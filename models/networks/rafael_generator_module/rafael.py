import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.networks.rafael_generator_module.rafael_trans_tools import to_2tuple, to_ntuple
from models.networks.rafael_generator_module.rafael_transformer_components import Mlp, Attention, DropPath



def split_int(num):
    """Split an integer into 2 integers evenly
  Args:
    num (int): The input integer
  Returns:
    num_1 (int)
    num_2 (int)
  """
    if num % 2 == 0:
        num_1 = num_2 = num // 2
    else:
        num_1 = num // 2
        num_2 = num_1 + 1
    return num_1, num_2


def unpad2D(input, pad):
    """Crop the input tensor according to pad.(Inverse operation for padding)
  Args:
    input (Tensor): (B, C, H, W)
    pad (Tuple of int): (left, right, top, bottom)
  Returns:
    output (Tensor): (B, C, new_H, new_W)
  """
    pad_W_left, pad_W_right, pad_H_top, pad_H_bottom = pad
    if pad_H_top == 0 and pad_H_bottom == 0 and not (pad_W_left == 0 and pad_W_right == 0):
        output = input[:, :, :, pad_W_left:-pad_W_right]
    elif pad_W_left == 0 and pad_W_right == 0 and not (pad_H_top == 0 and pad_H_bottom == 0):
        output = input[:, :, pad_H_top:-pad_H_bottom, :]
    elif pad_H_top == 0 and pad_H_bottom == 0 and pad_W_left == 0 and pad_W_right == 0:
        output = input
    else:
        output = input[:, :, pad_H_top:-pad_H_bottom, pad_W_left:-pad_W_right]
    return output


def seq_padding(x, dividable_size, input_resolution, pad_mode='constant'):
    """Padding for sequential data
  Args:
    x (Tensor): (B, L, C)
    dividable_size (Tuple | int): dividable size (div_H, div_W)
    input_resolution (Tuple): resolution of x (H_orig, W_orig)
  Returns:
    x_padded (Tensor): (B, new_L, C) where new_L = H_padded * W_padded
    output_resolution (Tuple): (H_padded, W_padded)
    pad_dims (Tuple of int): (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
  """
    H_orig, W_orig = input_resolution
    B, L_orig, C = x.shape
    assert L_orig == H_orig * W_orig, f'seq_padding: Input L_orig {L_orig} does not match H_orig*W_orig {H_orig * W_orig} ({H_orig}x{W_orig})'

    dividable_size_H, dividable_size_W = to_2tuple(dividable_size)

    x_img_domain = x.permute(0, 2, 1).reshape(B, C, H_orig, W_orig)

    rema_H, rema_W = H_orig % dividable_size_H, W_orig % dividable_size_W

    pad_H_total = dividable_size_H - rema_H if rema_H != 0 else 0
    pad_W_total = dividable_size_W - rema_W if rema_W != 0 else 0

    pad_H_top, pad_H_bottom = split_int(pad_H_total)
    pad_W_left, pad_W_right = split_int(pad_W_total)

    pad_dims = (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)

    needs_padding = pad_H_total > 0 or pad_W_total > 0
    if needs_padding:
        x_padded_img_domain = F.pad(x_img_domain, pad_dims, pad_mode, 0)
    else:
        x_padded_img_domain = x_img_domain

    H_padded, W_padded = x_padded_img_domain.shape[-2:]
    x_padded_seq_domain = x_padded_img_domain.reshape(B, C, -1).permute(0, 2, 1)

    assert x_padded_seq_domain.shape[1] == H_padded * W_padded, \
        f"seq_padding: Padded L {x_padded_seq_domain.shape[1]} != H_padded*W_padded {H_padded * W_padded} ({H_padded}x{W_padded})"

    return x_padded_seq_domain, (H_padded, W_padded), pad_dims


def seq_unpad(x, padded_resolution, pad_dims):
    """Unpadding for sequential data
  Args:
    x (Tensor): (B, L_padded, C)
    padded_resolution (Tuple): (H_padded, W_padded)
    pad_dims (Tuple of int): (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
  Returns:
    x_unpadded (Tensor): (B, new_L_orig, C)
    output_resolution (Tuple): (H_orig, W_orig)
  """
    H_padded, W_padded = padded_resolution
    B, L_padded, C = x.shape
    assert L_padded == H_padded * W_padded, \
        f'seq_unpad: Input L_padded {L_padded} does not match H_padded*W_padded {H_padded * W_padded} ({H_padded}x{W_padded})'

    x_img_domain = x.permute(0, 2, 1).reshape(B, C, H_padded, W_padded)

    needs_unpadding = any(p > 0 for p in pad_dims)
    if needs_unpadding:
        x_unpadded_img_domain = unpad2D(x_img_domain, pad=pad_dims)
    else:
        x_unpadded_img_domain = x_img_domain

    H_orig, W_orig = x_unpadded_img_domain.shape[-2:]
    x_unpadded_seq_domain = x_unpadded_img_domain.reshape(B, C, -1).permute(0, 2, 1)

    assert x_unpadded_seq_domain.shape[1] == H_orig * W_orig, \
        f"seq_unpad: Unpadded L {x_unpadded_seq_domain.shape[1]} != H_orig*W_orig {H_orig * W_orig} ({H_orig}x{W_orig})"

    return x_unpadded_seq_domain, (H_orig, W_orig)


########################################## Basic components ##########################################

def window_partition(x, window_size):
    win_H, win_W = to_2tuple(window_size)
    B, H, W, C = x.shape
    assert H % win_H == 0 and W % win_W == 0, \
        f"window_partition: H({H}) or W({W}) not divisible by window_size ({win_H}, {win_W}). This should be handled by padding."
    n_win_H = H // win_H
    n_win_W = W // win_W
    x = x.view(B, n_win_H, win_H, n_win_W, win_W, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_H, win_W, C)
    return windows


def window_reverse(windows, window_size, H, W):
    win_H, win_W = to_2tuple(window_size)
    assert H % win_H == 0 and W % win_W == 0, \
        f"window_reverse: H({H}) or W({W}) not divisible by window_size ({win_H}, {win_W})."
    n_win_H = H // win_H
    n_win_W = W // win_W
    num_total_windows = n_win_H * n_win_W
    assert windows.shape[0] % num_total_windows == 0, "window_reverse: Incorrect number of windows."
    B = windows.shape[0] // num_total_windows
    x = windows.view(B, n_win_H, n_win_W, win_H, win_W, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed_Kai(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.flatten = flatten
        self.proj = nn.Conv2d(self.in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H_img, W_img = x.shape
        assert C == self.in_chans, 'Input image need to have same numbers of channels with the initialed.'
        x_projected = self.proj(x)
        H_feat, W_feat = x_projected.shape[2], x_projected.shape[3]
        if self.flatten:
            x_flattened = x_projected.flatten(2).transpose(1, 2)
        else:
            x_flattened = x_projected
        x_normed = self.norm(x_flattened)
        return x_normed, (H_feat, W_feat)


class PatchMerging_Kai(nn.Module):
    def __init__(self, input_resolution, d_model, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.d_model = d_model
        self.reduction = nn.Linear(4 * d_model, 2 * d_model, bias=False)
        self.norm = norm_layer(4 * d_model)

    def forward(self, x_tuple):
        current_tensor = x_tuple[0]
        arbitrary_input = x_tuple[1]

        if arbitrary_input:
            H_feat, W_feat = x_tuple[2]
        else:
            H_feat, W_feat = self.input_resolution

        B, L_feat, C_feat = current_tensor.shape
        assert L_feat == H_feat * W_feat, \
            f"PatchMerging_Kai: Input L_feat {L_feat} does not match H_feat*W_feat {H_feat * W_feat} ({H_feat}x{W_feat}). Arbitrary: {arbitrary_input}. Expected H,W by PMK for fixed: {self.input_resolution}"

        x_img_domain = current_tensor.view(B, H_feat, W_feat, C_feat)
        H_eff_for_merge, W_eff_for_merge = H_feat, W_feat
        temp_x_img_domain = x_img_domain

        if H_feat % 2 != 0:
            temp_x_img_domain = temp_x_img_domain[:, 0:-1, :, :]
            H_eff_for_merge -= 1
        if W_feat % 2 != 0:
            temp_x_img_domain = temp_x_img_domain[:, :, 0:-1, :]
            W_eff_for_merge -= 1

        x0 = temp_x_img_domain[:, 0::2, 0::2, :]
        x1 = temp_x_img_domain[:, 1::2, 0::2, :]
        x2 = temp_x_img_domain[:, 0::2, 1::2, :]
        x3 = temp_x_img_domain[:, 1::2, 1::2, :]

        x_cat = torch.cat([x0, x1, x2, x3], -1)
        H_merged, W_merged = x_cat.shape[1], x_cat.shape[2]
        x_reshaped_seq = x_cat.view(B, -1, 4 * C_feat)
        assert x_reshaped_seq.shape[
                   1] == H_merged * W_merged, "PatchMerging_Kai: Logic error in patch count after cat and reshape."
        normed_x = self.norm(x_reshaped_seq)
        reduced_x = self.reduction(normed_x)
        return reduced_x, arbitrary_input, (H_merged, W_merged)


class WindowAttention_Kai(nn.Module):
    def __init__(self, d_model, window_size, nhead, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
        self.window_size_H, self.window_size_W = to_2tuple(window_size)
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size_H - 1) * (2 * self.window_size_W - 1), nhead)
        )
        coords_h = torch.arange(self.window_size_H)
        coords_w = torch.arange(self.window_size_W)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size_H - 1
        relative_coords[:, :, 1] += self.window_size_W - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_W - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, feat_resolution_shape):
        H_feat, W_feat = feat_resolution_shape
        B, N_feat, C = x.size()
        assert N_feat == H_feat * W_feat, \
            f"WindowAttention_Kai: Input N_feat {N_feat} != H_feat*W_feat {H_feat * W_feat} ({H_feat}x{W_feat})"
        assert H_feat % self.window_size_H == 0 and W_feat % self.window_size_W == 0, \
            f"WindowAttention_Kai: H_feat({H_feat}) or W_feat({W_feat}) not divisible by window_size ({self.window_size_H},{self.window_size_W})."
        x_img_domain = x.reshape(B, H_feat, W_feat, C)
        x_windows = window_partition(x_img_domain, (self.window_size_H, self.window_size_W))
        x_windows_seq = x_windows.reshape(-1, self.window_size_H * self.window_size_W, C)
        B_win, N_win, _ = x_windows_seq.shape
        qkv = self.qkv(x_windows_seq).reshape(B_win, N_win, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N_win, N_win, self.nhead)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn_out_win_seq = (attn @ v).transpose(1, 2).reshape(B_win, N_win, C)
        x_projected_win_seq = self.proj(x_attn_out_win_seq)
        x_dropped_win_seq = self.proj_drop(x_projected_win_seq)
        x_dropped_windows = x_dropped_win_seq.reshape(-1, self.window_size_H, self.window_size_W, C)
        x_img_domain_reconstructed = window_reverse(x_dropped_windows, (self.window_size_H, self.window_size_W), H_feat,
                                                    W_feat)
        x_final_seq = x_img_domain_reconstructed.reshape(B, N_feat, C)
        assert x_final_seq.shape[1] == N_feat, "WindowAttention_Kai: Output patch count mismatch."
        return x_final_seq


class LePEAttention(nn.Module):
    def __init__(self, d_model, nhead=8, strip_width=7, is_vertical=False, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.d_model = d_model
        self.strip_width = strip_width
        self.is_vertical = is_vertical
        self.attn = Attention(
            d_model=d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )

    def forward(self, x, feat_resolution_shape):
        H_feat, W_feat = feat_resolution_shape
        B, N_feat, C = x.size()
        assert N_feat == H_feat * W_feat, \
            f"LePEAttention Input: x patches {N_feat} != H_feat*W_feat {H_feat * W_feat} ({H_feat}x{W_feat})"
        x_img_domain = x.reshape(B, H_feat, W_feat, C)
        current_strip_window_H, current_strip_window_W = 0, 0
        num_patches_per_strip = 0
        if self.is_vertical:
            assert W_feat % self.strip_width == 0, \
                f"LePEAttention (vertical): W_feat({W_feat}) not divisible by strip_width ({self.strip_width})."
            current_strip_window_H, current_strip_window_W = H_feat, self.strip_width
            x_strips = window_partition(x_img_domain, (current_strip_window_H, current_strip_window_W))
            num_patches_per_strip = H_feat * self.strip_width
        else:
            assert H_feat % self.strip_width == 0, \
                f"LePEAttention (horizontal): H_feat({H_feat}) not divisible by strip_width ({self.strip_width})."
            current_strip_window_H, current_strip_window_W = self.strip_width, W_feat
            x_strips = window_partition(x_img_domain, (current_strip_window_H, current_strip_window_W))
            num_patches_per_strip = self.strip_width * W_feat
        x_strips_seq = x_strips.reshape(-1, num_patches_per_strip, C)
        attn_out_strips_seq = self.attn(x_strips_seq)
        attn_out_strips = attn_out_strips_seq.reshape(-1, current_strip_window_H, current_strip_window_W, C)
        x_img_domain_reconstructed = window_reverse(attn_out_strips, (current_strip_window_H, current_strip_window_W),
                                                    H_feat, W_feat)
        x_final_seq = x_img_domain_reconstructed.reshape(B, N_feat, C)
        assert x_final_seq.shape[1] == N_feat, "LePEAttention: Output patch count mismatch."
        return x_final_seq


class LePEAttentionBlock(nn.Module):
    def __init__(self, d_model, input_resolution, nhead=8, strip_width=7,
                 mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.input_resolution_fixed = to_2tuple(input_resolution)  # input_resolution cho fixed path
        self.strip_width = strip_width
        self.norm1 = norm_layer(d_model)
        self.attn1 = LePEAttention(
            d_model=d_model, nhead=nhead, strip_width=strip_width, is_vertical=False,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.attn2 = LePEAttention(
            d_model=d_model, nhead=nhead, strip_width=strip_width, is_vertical=True,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.attn3 = WindowAttention_Kai(
            d_model=d_model, window_size=(strip_width * 2, strip_width * 2), nhead=nhead,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)

    def forward(self, x_tuple):
        current_tensor = x_tuple[0]
        arbitrary_input = x_tuple[1]
        H_orig, W_orig = x_tuple[2]

        assert current_tensor.shape[1] == H_orig * W_orig, \
            f"LePEAttentionBlock Input: tensor patches {current_tensor.shape[1]} != H_orig*W_orig {H_orig * W_orig} ({H_orig}x{W_orig})"

        dividable_size_for_padding = self.strip_width * 2

        # --- SỬA ĐỔI Ở ĐÂY ---
        # Luôn thực hiện padding nếu H_orig hoặc W_orig không chia hết, bất kể arbitrary_input
        # Điều này đảm bảo các lớp attention bên trong nhận được kích thước chia hết.
        needs_padding_for_block = (H_orig % dividable_size_for_padding != 0) or \
                                  (W_orig % dividable_size_for_padding != 0) or \
                                  (H_orig % self.strip_width != 0) or \
                                  (W_orig % self.strip_width != 0)  # Thêm kiểm tra cho strip_width đơn lẻ

        if needs_padding_for_block:
            # Tính toán dividable_size tổng thể cho cả strip_width và strip_width*2
            # lcm(strip_width, strip_width*2) = strip_width*2
            actual_dividable_size = self.strip_width * 2

            padded_x, (H_padded, W_padded), pad_dims = seq_padding(
                current_tensor,
                dividable_size=actual_dividable_size,  # Đảm bảo chia hết cho cả sw và sw*2
                input_resolution=(H_orig, W_orig),
                pad_mode='constant'
            )
        else:  # Không cần padding, H_orig, W_orig đã chia hết
            H_padded, W_padded = H_orig, W_orig
            padded_x = current_tensor
            pad_dims = (0, 0, 0, 0)
        # -----------------------

        shortcut = padded_x
        normed_x = self.norm1(padded_x)

        x1 = self.attn1(normed_x, feat_resolution_shape=(H_padded, W_padded))
        x2 = self.attn2(normed_x, feat_resolution_shape=(H_padded, W_padded))
        x3 = self.attn3(normed_x, feat_resolution_shape=(H_padded, W_padded))

        assert x1.shape == normed_x.shape, f"Shape mismatch after attn1: {x1.shape} vs {normed_x.shape}"
        assert x2.shape == normed_x.shape, f"Shape mismatch after attn2: {x2.shape} vs {normed_x.shape}"
        assert x3.shape == normed_x.shape, f"Shape mismatch after attn3: {x3.shape} vs {normed_x.shape}"

        q_for_combine = normed_x.unsqueeze(dim=2)
        k_v_for_combine = torch.stack([normed_x, x1, x2, x3], dim=2)
        attn_on_attn_results = (q_for_combine @ k_v_for_combine.transpose(-1, -2)).softmax(dim=-1)
        x_after_combine_method = (attn_on_attn_results @ k_v_for_combine).squeeze(dim=2)

        assert x_after_combine_method.shape == shortcut.shape, \
            f"Shape mismatch after combining attentions: {x_after_combine_method.shape} vs {shortcut.shape}"

        x_residual_after_attn = shortcut + self.drop_path(x_after_combine_method)
        x_after_ffn = x_residual_after_attn + self.drop_path(self.mlp(self.norm2(x_residual_after_attn)))

        output_tensor_before_unpad = x_after_ffn
        final_output_resolution_before_unpad = (H_padded, W_padded)

        # Luôn unpad nếu có padding đã được áp dụng (pad_dims != (0,0,0,0))
        if needs_padding_for_block:  # Hoặc đơn giản là if any(p > 0 for p in pad_dims):
            unpadded_x, (H_final_orig, W_final_orig) = seq_unpad(
                output_tensor_before_unpad,
                padded_resolution=(H_padded, W_padded),  # Kích thước của tensor trước khi unpad
                pad_dims=pad_dims
            )
            # H_final_orig, W_final_orig phải bằng H_orig, W_orig ban đầu
            assert H_final_orig == H_orig and W_final_orig == W_orig, \
                f"Unpadding error: Original ({H_orig},{W_orig}) vs Unpadded ({H_final_orig},{W_final_orig})"
            output_tensor_final = unpadded_x
            final_output_resolution = (H_final_orig, W_final_orig)
        else:
            output_tensor_final = output_tensor_before_unpad
            final_output_resolution = final_output_resolution_before_unpad  # (H_orig, W_orig)

        return (output_tensor_final, arbitrary_input, final_output_resolution)


class BasicLayer_SA(nn.Module):
    def __init__(self, d_model, input_resolution, depth, nhead, strip_width,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.d_model = d_model
        self.input_resolution_fixed = to_2tuple(input_resolution)
        self.depth = depth
        if isinstance(strip_width, list):
            assert len(strip_width) == depth, "strip_width list length must match layer depth"
            self.block_strip_widths = strip_width
        else:
            self.block_strip_widths = [strip_width] * depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            LePEAttentionBlock(
                d_model=d_model,
                input_resolution=self.input_resolution_fixed,
                nhead=nhead,
                strip_width=self.block_strip_widths[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(self.depth)
        ])
        if downsample is not None:
            self.downsample = downsample(self.input_resolution_fixed, d_model=d_model, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x_tuple_input):
        _tensor_in_bsl, _arb_in_bsl, (_H_in_bsl, _W_in_bsl) = x_tuple_input
        if not _arb_in_bsl:
            assert (_H_in_bsl, _W_in_bsl) == self.input_resolution_fixed, \
                f"BasicLayer_SA (Fixed): Input tuple H,W ({_H_in_bsl},{_W_in_bsl}) != self.input_resolution_fixed {self.input_resolution_fixed}"
            assert _tensor_in_bsl.shape[1] == self.input_resolution_fixed[0] * self.input_resolution_fixed[1], \
                f"BasicLayer_SA (Fixed): Input tensor patches {_tensor_in_bsl.shape[1]} != expected {self.input_resolution_fixed[0] * self.input_resolution_fixed[1]}"
        else:
            assert _tensor_in_bsl.shape[1] == _H_in_bsl * _W_in_bsl, \
                f"BasicLayer_SA (Arbitrary): Input tensor patches {_tensor_in_bsl.shape[1]} != H*W from tuple {_H_in_bsl * _W_in_bsl}"

        current_processing_tuple = x_tuple_input
        for blk in self.blocks:
            if self.use_checkpoint:
                current_processing_tuple = blk(current_processing_tuple)

            else:
                current_processing_tuple = blk(current_processing_tuple)

        if self.downsample is not None:
            current_processing_tuple = self.downsample(current_processing_tuple)
        return current_processing_tuple


class RAFAEL(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], nhead=[3, 6, 12, 24],
                 strip_width=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()
        self.img_size_fixed = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_layers = len(depths)
        if isinstance(strip_width, (list, tuple)):
            assert len(
                strip_width) == self.num_layers, "strip_width list/tuple length must match num_layers (depths length)"
            self.layer_strip_widths = list(strip_width)
        else:
            self.layer_strip_widths = [strip_width] * self.num_layers
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed_Kai(
            patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patches_resolution_fixed = (self.img_size_fixed[0] // self.patch_size[0],
                                         self.img_size_fixed[1] // self.patch_size[1])
        self.num_patches_fixed = self.patches_resolution_fixed[0] * self.patches_resolution_fixed[1]
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_fixed, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        current_d_model = self.embed_dim
        current_H_fixed = self.patches_resolution_fixed[0]
        current_W_fixed = self.patches_resolution_fixed[1]
        for i in range(self.num_layers):
            layer_input_resolution_fixed = (current_H_fixed, current_W_fixed)
            layer = BasicLayer_SA(
                d_model=current_d_model, input_resolution=layer_input_resolution_fixed,
                depth=depths[i], nhead=nhead[i], strip_width=self.layer_strip_widths[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer,
                downsample=PatchMerging_Kai if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            if (i < self.num_layers - 1):
                current_d_model *= 2
                current_H_fixed //= 2
                current_W_fixed //= 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x_img_tensor, arbitrary_input_flag, initial_img_resolution_tuple_unused):
        x_after_embed, (H_feat, W_feat) = self.patch_embed(x_img_tensor)
        if self.ape:
            if not arbitrary_input_flag and (H_feat * W_feat == self.num_patches_fixed):
                x_after_embed = x_after_embed + self.absolute_pos_embed
            elif arbitrary_input_flag:
                pass
        x_after_pos_drop = self.pos_drop(x_after_embed)
        current_processing_tuple = (x_after_pos_drop, arbitrary_input_flag, (H_feat, W_feat))
        for layer in self.layers:
            current_processing_tuple = layer(current_processing_tuple)
        return current_processing_tuple

    def forward(self, x_img, arbitrary_input=False):
        img_H_orig, img_W_orig = x_img.shape[2], x_img.shape[3]
        output_tuple = self.forward_features(x_img, arbitrary_input, (img_H_orig, img_W_orig))
        return output_tuple