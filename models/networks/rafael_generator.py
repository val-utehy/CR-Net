import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
from models.networks.base_network import BaseNetwork
from models.networks.rafael_generator_module.rafael import RAFAEL as RAFAEL_Content_Encoder
from .time_utils import get_day_night_weights


class AdaINLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_norm = None

    def forward(self, content_feat, style_gamma, style_beta):
        if content_feat is None or style_gamma is None or style_beta is None: return content_feat
        if content_feat.numel() == 0: return content_feat
        if self.instance_norm is None or self.instance_norm.num_features != content_feat.size(1):
            if content_feat.size(1) == 0: return content_feat
            self.instance_norm = nn.InstanceNorm2d(content_feat.size(1), affine=False, track_running_stats=False).to(
                content_feat.device)
        normalized_content = self.instance_norm(content_feat)
        return style_gamma * normalized_content + style_beta


class AdaINResBlockWithLatent(nn.Module):
    def __init__(self, channels, latent_dim, use_reflection_pad=True):
        super().__init__()
        self.use_reflection_pad = use_reflection_pad
        self.mlp_style1 = nn.Linear(latent_dim, channels * 2)
        self.pad1 = nn.ReflectionPad2d(1) if use_reflection_pad else nn.Identity()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.mlp_style2 = nn.Linear(latent_dim, channels * 2)
        self.pad2 = nn.ReflectionPad2d(1) if use_reflection_pad else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.adain1 = AdaINLayer()
        self.adain2 = AdaINLayer()

    def forward(self, x, z_style):
        skip_connection = x
        style_params1 = self.mlp_style1(z_style)
        gamma1 = style_params1[:, :x.size(1)].unsqueeze(-1).unsqueeze(-1)
        beta1 = style_params1[:, x.size(1):].unsqueeze(-1).unsqueeze(-1)
        stylized_x = self.adain1(x, gamma1, beta1)
        h = self.pad1(stylized_x)
        h = self.relu1(self.conv1(h))
        style_params2 = self.mlp_style2(z_style)
        gamma2 = style_params2[:, :h.size(1)].unsqueeze(-1).unsqueeze(-1)
        beta2 = style_params2[:, h.size(1):].unsqueeze(-1).unsqueeze(-1)
        stylized_h = self.adain2(h, gamma2, beta2)
        h = self.pad2(stylized_h)
        h = self.conv2(h)
        return F.relu(h + skip_connection)


class AdaINSynthesisNetworkWithLatent(BaseNetwork):
    def __init__(self, opt, content_encoder_output_dim, initial_content_spatial_res, style_latent_dim):
        super().__init__()
        self.opt = opt
        current_channels = content_encoder_output_dim
        self.initial_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                          nn.Conv2d(content_encoder_output_dim, current_channels, kernel_size=3,
                                                    padding=0), nn.ReLU())
        res_blocks_list = []
        for _ in range(opt.synth_num_res_blocks): res_blocks_list.append(
            AdaINResBlockWithLatent(current_channels, style_latent_dim))
        self.res_blocks = nn.ModuleList(res_blocks_list)
        upsample_blocks_list = []
        target_h, current_h = opt.rafael_img_size, initial_content_spatial_res[0]
        num_upsamples_needed = 3
        if current_h > 0 and target_h > current_h:
            ratio = target_h / current_h
            if ratio == int(ratio) and (int(ratio) & (int(ratio) - 1)) == 0: num_upsamples_needed = int(
                math.log2(ratio))
        for i in range(num_upsamples_needed):
            out_channels = current_channels // 2 if current_channels > opt.ngf * 2 else opt.ngf
            if out_channels < opt.ngf and opt.ngf > 0: out_channels = opt.ngf
            if out_channels <= 0: out_channels = opt.ngf if opt.ngf > 0 else 3
            upsample_blocks_list.extend([nn.Upsample(scale_factor=2, mode='nearest'), nn.ReflectionPad2d(1),
                                         nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=0), nn.ReLU(),
                                         AdaINResBlockWithLatent(out_channels, style_latent_dim)])
            current_channels = out_channels
        self.upsample_blocks = nn.ModuleList(upsample_blocks_list)
        self.final_conv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(current_channels, 3, kernel_size=7, padding=0),
                                        nn.Tanh())

    def forward(self, content_features_seq, content_spatial_res_tuple, z_style1, z_style2=None, mixing_prob=0.0,
                arbitrary_input=False):
        B, L, C_enc = content_features_seq.shape
        H_feat, W_feat = content_spatial_res_tuple
        x = content_features_seq.permute(0, 2, 1).contiguous().view(B, C_enc, H_feat, W_feat)
        x = self.initial_conv(x)
        num_adain_blocks = len(self.res_blocks) + len(
            [op for op in self.upsample_blocks if isinstance(op, AdaINResBlockWithLatent)])
        mixing_point = num_adain_blocks + 1
        if z_style2 is not None and mixing_prob > 0 and random.random() < mixing_prob: mixing_point = torch.randint(1,
                                                                                                                    num_adain_blocks,
                                                                                                                    (
                                                                                                                        1,)).item()
        current_adain_block_idx = 0
        for block in self.res_blocks:
            z_current = z_style2 if current_adain_block_idx >= mixing_point else z_style1
            x = block(x, z_current)
            current_adain_block_idx += 1
        upsample_op_idx = 0
        while upsample_op_idx < len(self.upsample_blocks):
            x = self.upsample_blocks[upsample_op_idx](x);
            upsample_op_idx += 1
            x = self.upsample_blocks[upsample_op_idx](x);
            upsample_op_idx += 1
            x = self.upsample_blocks[upsample_op_idx](x);
            upsample_op_idx += 1
            x = self.upsample_blocks[upsample_op_idx](x);
            upsample_op_idx += 1
            res_block = self.upsample_blocks[upsample_op_idx]
            z_current = z_style2 if current_adain_block_idx >= mixing_point else z_style1
            x = res_block(x, z_current)
            upsample_op_idx += 1
            current_adain_block_idx += 1
        return self.final_conv(x)


class RafaelGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--rafael_img_size', type=int, default=256)
        parser.add_argument('--rafael_patch_size', type=int, default=2)
        parser.add_argument('--rafael_embed_dim', type=int, default=192)
        parser.add_argument('--rafael_depths', type=str, default='2,2,2')
        parser.add_argument('--rafael_nheads', type=str, default='3,6,12')
        parser.add_argument('--rafael_strip_widths', type=str, default='2,4,7')
        parser.add_argument('--rafael_use_checkpoint', action='store_true')
        parser.add_argument('--style_enc_latent_dim', type=int, default=256, help='Output latent z dimension')
        parser.add_argument('--synth_num_res_blocks', type=int, default=4)
        if is_train:
            parser.add_argument('--style_mixing_prob', type=float, default=0.9)

        parser.add_argument('--light_map_channels', type=int, default=1,
                            help='Số kênh cho bản đồ hướng sáng (1 hoặc 2).')
        parser.add_argument('--add_phi_to_latent', action='store_true', help='Thêm (cos(phi), sin(phi)) vào z_style')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        rafael_depths = [int(d) for d in opt.rafael_depths.split(',') if d.strip()]
        rafael_nheads = [int(h) for h in opt.rafael_nheads.split(',') if h.strip()]
        rafael_strip_widths_str = opt.rafael_strip_widths.split(',')
        if len(rafael_strip_widths_str) == 1 and rafael_strip_widths_str[0].strip():
            rafael_strip_widths = [int(rafael_strip_widths_str[0])] * len(rafael_depths)
        else:
            rafael_strip_widths = [int(sw) for sw in rafael_strip_widths_str if sw.strip()]

        content_encoder_in_chans = 3 + opt.light_map_channels

        self.rafael_content_encoder = RAFAEL_Content_Encoder(
            img_size=opt.rafael_img_size, patch_size=opt.rafael_patch_size,
            in_chans=content_encoder_in_chans,
            embed_dim=opt.rafael_embed_dim, depths=rafael_depths, nhead=rafael_nheads,
            strip_width=rafael_strip_widths, use_checkpoint=opt.rafael_use_checkpoint
        )

        num_downsamples_in_enc = len(rafael_depths) - 1
        content_encoder_output_dim = opt.rafael_embed_dim * (2 ** num_downsamples_in_enc)

        initial_spatial_res_h = opt.rafael_img_size // opt.rafael_patch_size
        initial_spatial_res_w = opt.rafael_img_size // opt.rafael_patch_size
        self.content_encoder_spatial_res = (
            initial_spatial_res_h // (2 ** num_downsamples_in_enc),
            initial_spatial_res_w // (2 ** num_downsamples_in_enc)
        )

        self.latent_dim = opt.style_enc_latent_dim
        synthesis_latent_dim = self.latent_dim + 2 if opt.add_phi_to_latent else self.latent_dim
        self.z_day = nn.Parameter(torch.randn(1, self.latent_dim))
        self.z_night = nn.Parameter(torch.randn(1, self.latent_dim))

        self.synthesis_network = AdaINSynthesisNetworkWithLatent(
            opt=opt,
            content_encoder_output_dim=content_encoder_output_dim,
            initial_content_spatial_res=self.content_encoder_spatial_res,
            style_latent_dim=synthesis_latent_dim
        )

    def forward(self, content_image_with_light, w_day, w_night, cos_phi, sin_phi, phi2=None, arbitrary_input=False):
        current_device = content_image_with_light.device
        batch_size = content_image_with_light.size(0)

        f_c_tuple = self.rafael_content_encoder(content_image_with_light, arbitrary_input=arbitrary_input)
        content_features_seq, content_spatial_res = f_c_tuple[0], f_c_tuple[2]

        z_day_learned = self.z_day.expand(batch_size, -1)
        z_night_learned = self.z_night.expand(batch_size, -1)

        w_day1_expanded = w_day.view(-1, 1).expand_as(z_day_learned)
        w_night1_expanded = w_night.view(-1, 1).expand_as(z_night_learned)
        z_style1_base = w_day1_expanded * z_day_learned + w_night1_expanded * z_night_learned

        if self.opt.add_phi_to_latent:
            z_style1 = torch.cat([z_style1_base, cos_phi.view(-1, 1), sin_phi.view(-1, 1)], dim=1)
        else:
            z_style1 = z_style1_base

        z_style2 = None
        if phi2 is not None and self.training:
            phi2_tensor = phi2.to(current_device, dtype=z_day_learned.dtype)

            w_day2, w_night2 = get_day_night_weights(phi2_tensor, self.opt.daylight_curve_steepness)

            w_day2_expanded = w_day2.view(-1, 1).expand_as(z_day_learned)
            w_night2_expanded = w_night2.view(-1, 1).expand_as(z_night_learned)
            z_style2_base = w_day2_expanded * z_day_learned + w_night2_expanded * z_night_learned

            if self.opt.add_phi_to_latent:
                cos_phi2, sin_phi2 = torch.cos(phi2_tensor), torch.sin(phi2_tensor)
                z_style2 = torch.cat([z_style2_base, cos_phi2.view(-1, 1), sin_phi2.view(-1, 1)], dim=1)
            else:
                z_style2 = z_style2_base

        mixing_prob = self.opt.style_mixing_prob if self.training else 0.0
        i_cs = self.synthesis_network(content_features_seq, content_spatial_res, z_style1, z_style2, mixing_prob,
                                      arbitrary_input=arbitrary_input)

        generator_returns = {}

        if self.training:
            z_day_for_identity = z_day_learned
            z_night_for_identity = z_night_learned
            if self.opt.add_phi_to_latent:
                day_phi_components = torch.tensor([[-1.0, 0.0]], device=current_device).expand(batch_size, -1)
                night_phi_components = torch.tensor([[1.0, 0.0]], device=current_device).expand(batch_size, -1)
                z_day_for_identity = torch.cat([z_day_learned, day_phi_components], dim=1)
                z_night_for_identity = torch.cat([z_night_learned, night_phi_components], dim=1)

            i_cc = self.synthesis_network(content_features_seq, content_spatial_res, z_day_for_identity,
                                          mixing_prob=0.0, arbitrary_input=arbitrary_input)
            i_nn = self.synthesis_network(content_features_seq, content_spatial_res, z_night_for_identity,
                                          mixing_prob=0.0, arbitrary_input=arbitrary_input)
            generator_returns['identity_content'] = i_cc
            generator_returns['identity_night'] = i_nn

        return i_cs, generator_returns

