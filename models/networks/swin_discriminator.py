import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.nn import init

from models.networks.base_network import BaseNetwork


def _get_swin_d_init_func_helper(it_type, it_gain):
    """Helper function to get the initialization function."""

    def init_func_local(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:  # Handle InstanceNorm too
            if hasattr(m, 'weight') and m.weight is not None: init.normal_(m.weight.data, 1.0, it_gain)
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if it_type == 'normal':
                init.normal_(m.weight.data, 0.0, it_gain)
            elif it_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=it_gain)
            elif it_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif it_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=it_gain)
            elif it_type == 'none':
                pass
            else:
                raise NotImplementedError(f'initialization method [{it_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)

    return init_func_local


class ConditionEncoder(nn.Module):
    def __init__(self, input_nc=3, output_dim=None, num_downsamplings=3, ngf=64):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, num_downsamplings):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers.extend([
                nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult),
                nn.LeakyReLU(0.2, True)
            ])
        final_cnn_channels = ngf * mult
        if output_dim is not None and output_dim != final_cnn_channels:
            layers.append(nn.Conv2d(final_cnn_channels, output_dim, kernel_size=1, stride=1, padding=0))
            self.final_channels = output_dim
        else:
            self.final_channels = final_cnn_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        B_q, N_q, C_q = query.shape
        B_kv, N_kv, C_kv = key_value.shape
        assert C_q == C_kv, f"Query channels {C_q} and Key-Value channels {C_kv} must match"
        dim = C_q
        q = self.q_proj(query).reshape(B_q, N_q, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B_kv, N_kv, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B_kv, N_kv, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = CrossAttentionLayer(dim, num_heads, qkv_bias, attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()

    def forward(self, query_feat, condition_feat):
        attn_out = self.cross_attn(self.norm_q(query_feat), self.norm_kv(condition_feat))
        query_feat = query_feat + self.drop_path(attn_out)
        return query_feat


class SwinTransformerConditionalDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--swin_d_model_name', type=str, default='swinv2_tiny_window8_256.ms_in1k',
                            help="Name of the Swin model from timm.")
        parser.add_argument('--swin_d_pretrained', action='store_true', help="Use pretrained Swin backbone.")
        parser.add_argument('--swin_d_output_style', type=str, default='patch', choices=['patch', 'global'],
                            help="Output style: 'patch' for patch-wise scores, 'global' for a single score.")
        parser.add_argument('--swin_d_num_conv_head', type=int, default=2,
                            help="Number of conv layers in the patch-output head.")
        parser.add_argument('--swin_d_use_sigmoid', action='store_true', help="Apply sigmoid to the final D output.")
        parser.add_argument('--cond_enc_input_nc', type=int, default=3, help='Input channels for condition encoder.')
        parser.add_argument('--cond_enc_ngf', type=int, default=64,
                            help='Base number of filters for condition encoder.')
        parser.add_argument('--cond_enc_num_downs', type=int, default=3,
                            help='Number of downsampling layers in condition encoder.')
        parser.add_argument('--d_cross_attn_num_heads', type=int, default=4,
                            help='Number of heads for cross attention in D.')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.output_style = opt.swin_d_output_style
        swin_input_nc = 3

        self.user_target_input_size = (opt.crop_size, opt.crop_size)

        self.backbone = timm.create_model(
            opt.swin_d_model_name,
            pretrained=opt.swin_d_pretrained,
            num_classes=0,
            in_chans=swin_input_nc,
            img_size=self.user_target_input_size
        )
        self.expected_input_size = self.user_target_input_size

        self.final_swin_embed_dim = self.backbone.num_features
        assert self.final_swin_embed_dim > 0, "Could not determine backbone's feature dimension."

        with torch.no_grad():
            dummy_b_in = torch.randn(1, swin_input_nc, self.expected_input_size[0], self.expected_input_size[1])
            dummy_b_feat_raw = self.backbone.forward_features(dummy_b_in)

            # Timm models for Swin output (B, H, W, C) or (B, L, C)
            if dummy_b_feat_raw.dim() == 4 and dummy_b_feat_raw.shape[-1] == self.final_swin_embed_dim:  # NHWC
                self.H_feat_approx = dummy_b_feat_raw.shape[1]
                self.W_feat_approx = dummy_b_feat_raw.shape[2]
            elif dummy_b_feat_raw.dim() == 3 and dummy_b_feat_raw.shape[-1] == self.final_swin_embed_dim:  # BLC
                L_approx = dummy_b_feat_raw.shape[1]
                # Infer H, W from L, assuming a square-like feature map
                self.H_feat_approx = self.W_feat_approx = int(L_approx ** 0.5)
                if self.H_feat_approx * self.W_feat_approx != L_approx:
                    print(f"Warning: Cannot infer square feature map size from L={L_approx}. Fallback needed.")
            else:
                raise RuntimeError(
                    f"Unexpected Swin backbone output shape: {dummy_b_feat_raw.shape}. Expected NHWC or BLC format.")

        self.condition_encoder = ConditionEncoder(
            input_nc=opt.cond_enc_input_nc,
            output_dim=self.final_swin_embed_dim,
            num_downsamplings=opt.cond_enc_num_downs,
            ngf=opt.cond_enc_ngf)
        assert self.condition_encoder.final_channels == self.final_swin_embed_dim, f"ConditionEncoder output_dim ({self.condition_encoder.final_channels}) mismatch with Swin embed_dim ({self.final_swin_embed_dim})"

        self.final_cross_attention_block = CrossAttentionBlock(
            dim=self.final_swin_embed_dim,
            num_heads=opt.d_cross_attn_num_heads)

        self.num_features_for_head = self.final_swin_embed_dim

        if self.output_style == 'patch':
            patch_head_layers = []
            current_channels_head = self.num_features_for_head
            for i in range(opt.swin_d_num_conv_head - 1):
                out_ch_head = current_channels_head // 2
                patch_head_layers.extend([
                    nn.Conv2d(current_channels_head, out_ch_head, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(out_ch_head),
                    nn.LeakyReLU(0.2, True)
                ])
                current_channels_head = out_ch_head
            patch_head_layers.append(nn.Conv2d(current_channels_head, 1, kernel_size=3, padding=1))
            if opt.swin_d_use_sigmoid: patch_head_layers.append(nn.Sigmoid())
            self.head = nn.Sequential(*patch_head_layers)
        elif self.output_style == 'global':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            fc1_out = self.num_features_for_head // 2
            self.head = nn.Sequential(
                nn.Linear(self.num_features_for_head, fc1_out),
                nn.LeakyReLU(0.2, True),
                nn.Linear(fc1_out, 1)
            )
            if opt.swin_d_use_sigmoid: self.head.add_module("sigmoid", nn.Sigmoid())

    def init_weights(self, init_type='xavier', gain=0.02):
        pass

    def forward(self, main_img, condition_img):
        if main_img.shape[2:] != self.expected_input_size:
            main_img = F.interpolate(main_img, size=self.expected_input_size, mode='bilinear', align_corners=False)
        if condition_img.shape[2:] != self.expected_input_size:
            condition_img = F.interpolate(condition_img, size=self.expected_input_size, mode='bilinear',
                                          align_corners=False)

        features_main_raw = self.backbone.forward_features(main_img)

        if features_main_raw.dim() == 4:  # NHWC format
            B, Hf, Wf, C_swin = features_main_raw.shape
            features_main_seq = features_main_raw.reshape(B, Hf * Wf, C_swin)
            actual_H_feat_main, actual_W_feat_main = Hf, Wf
        elif features_main_raw.dim() == 3:  # BLC format
            features_main_seq = features_main_raw
            actual_H_feat_main, actual_W_feat_main = self.H_feat_approx, self.W_feat_approx
        else:
            raise ValueError(f"Unexpected Swin output shape: {features_main_raw.shape}")

        cond_feat_map = self.condition_encoder(condition_img)
        cond_feat_seq = cond_feat_map.flatten(2).transpose(1, 2).contiguous()

        fused_features = self.final_cross_attention_block(features_main_seq, cond_feat_seq)

        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            final_features_for_head = self.backbone.norm(fused_features)
        else:
            final_features_for_head = fused_features

        output = None
        features_for_loss = None
        B_ff, L_ff, C_ff = final_features_for_head.shape

        if self.output_style == 'patch':
            features_2d = final_features_for_head.permute(0, 2, 1).reshape(B_ff, C_ff, actual_H_feat_main,
                                                                           actual_W_feat_main)
            output = self.head(features_2d)
            features_for_loss = features_2d
        elif self.output_style == 'global':
            pooled_features = self.global_pool(final_features_for_head.transpose(1, 2)).squeeze(-1)
            output = self.head(pooled_features)
            features_for_loss = pooled_features

        if output is None:
            raise RuntimeError("Discriminator output failed to compute.")

        return features_for_loss, output