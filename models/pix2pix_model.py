import torch
import models.networks as networks
from models.networks.loss import VGG16, SpatialCorrelativeLoss, SSIMLoss, LaplacianLoss
from models.networks.time_utils import get_day_night_weights
import util.util as util
import torch.nn as nn
import os
import torch.nn.functional as F
import math
import random


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (batch_size * c * h * w)

class ThresholdedL1Loss(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, input_tensor):
        avg_brightness = torch.mean(input_tensor, dim=[1, 2, 3])
        loss = F.relu(avg_brightness - self.threshold)
        return torch.mean(loss)


def hour_to_rad(hour):
    return (hour / 12.0) * math.pi


def create_light_direction_map(phi, size, num_channels, device, w_day):
    B = phi.shape[0]
    H, W = size
    x_coords = torch.linspace(-1, 1, W, device=device)
    y_coords = torch.linspace(-1, 1, H, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)
    phi_adjusted = phi - math.pi / 2
    light_vec_x = torch.cos(phi_adjusted).view(B, 1, 1)
    light_vec_y = torch.sin(phi_adjusted).view(B, 1, 1)

    directional_intensity = w_day.view(B, 1, 1)

    if num_channels == 1:
        light_map = (x_grid * light_vec_x + y_grid * light_vec_y) * directional_intensity
        return light_map.unsqueeze(1)
    elif num_channels == 2:
        light_map_x = x_grid * light_vec_x * directional_intensity
        light_map_y = y_grid * light_vec_y * directional_intensity
        return torch.stack([light_map_x, light_map_y], dim=1)
    else:
        raise ValueError("num_channels for light map must be 1 or 2")


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)

        parser.add_argument('--daylight_curve_steepness', type=float, default=0.75)
        parser.add_argument('--lambda_vgg_interp', type=float, default=15.0)
        parser.add_argument('--lambda_tv', type=float, default=5.0)

        parser.add_argument('--attn_layers', type=str, default='4,7,9')
        parser.add_argument('--patch_nums', type=float, default=128)
        parser.add_argument('--patch_size', type=int, default=32)
        parser.add_argument('--loss_mode', type=str, default='cos')
        parser.add_argument('--use_norm', action='store_true')
        parser.add_argument('--learned_attn', action='store_true', default=False)
        parser.add_argument('--T', type=float, default=0.07)
        parser.add_argument('--lambda_spatial', type=float, default=10.0)
        parser.add_argument('--lambda_G', type=float, default=1.0)
        if is_train:
            parser.add_argument('--night_loss_warmup_iters', type=int, default=10000)
            parser.add_argument('--use_thresholded_blackout', action='store_true')
            parser.add_argument('--blackout_threshold', type=float, default=-0.9)
            parser.add_argument('--lambda_thresholded_blackout', type=float, default=20.0)

            parser.add_argument('--lambda_content_focus', type=float, default=5.0)
            parser.add_argument('--lambda_continuity', type=float, default=10.0)
            parser.add_argument('--lambda_style', type=float, default=30.0)
            parser.add_argument('--lambda_identity', type=float, default=10.0)
            parser.add_argument('--lambda_style_recon', type=float, default=15.0)
            parser.add_argument('--lambda_r1', type=float, default=10.0)
            parser.add_argument('--lambda_latent_diversity', type=float, default=1.0)
            parser.add_argument('--use_patch_vgg_loss', action='store_true')
            parser.add_argument('--lambda_patch_vgg', type=float, default=10.0)
            parser.add_argument('--patch_vgg_num_patches', type=int, default=64)
            parser.add_argument('--patch_vgg_size', type=int, default=64)
            parser.add_argument('--lambda_ppl', type=float, default=2.0)
            parser.add_argument('--ppl_reg_every', type=int, default=4)
            parser.add_argument('--r1_reg_every', type=int, default=16)

            parser.add_argument('--lambda_laplacian', type=float, default=5.0)
            parser.add_argument('--lambda_ssim', type=float, default=1.0)

            if not hasattr(parser.parse_known_args()[0], 'lambda_z_reg'):
                parser.add_argument('--lambda_z_reg', type=float, default=0.0)
            if not hasattr(parser.parse_known_args()[0], 'lambda_night_content'):
                parser.add_argument('--lambda_night_content', type=float, default=0.0)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
        self.device = self._get_current_device_for_loss_init()
        self.netG, self.netD, self.netD2, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode,
                                                 tensor=torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor,
                                                 opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionIdentity = torch.nn.L1Loss()
            if (not opt.no_vgg_loss and opt.lambda_vgg > 0) or \
                    (hasattr(opt, 'lambda_style') and opt.lambda_style > 0) or \
                    (hasattr(opt, 'lambda_vgg_interp') and opt.lambda_vgg_interp > 0) or \
                    (hasattr(opt, 'lambda_night_content') and opt.lambda_night_content > 0) or \
                    (hasattr(opt, 'lambda_continuity') and opt.lambda_continuity > 0):
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
                if self.use_gpu(): self.criterionVGG.to(self.device)

            if opt.lambda_tv > 0:
                self.criterionTV = TVLoss().to(self.device)

            if opt.lambda_ssim > 0:
                self.criterionSSIM = SSIMLoss().to(self.device)
            if opt.lambda_laplacian > 0:
                self.criterionLaplacian = LaplacianLoss(self.device).to(self.device)

            if opt.use_thresholded_blackout:
                self.criterionThresholdedBlackout = ThresholdedL1Loss(threshold=opt.blackout_threshold).to(self.device)

            self.vgg_for_spatial = None
            self.criterionSpatial = None
            if opt.lambda_spatial > 0:
                self.vgg_for_spatial = VGG16().to(self.device)
                self.vgg_for_spatial.eval()
                for param in self.vgg_for_spatial.parameters(): param.requires_grad = False
                self.criterionSpatial = SpatialCorrelativeLoss(loss_mode=opt.loss_mode, patch_nums=opt.patch_nums,
                                                               patch_size=opt.patch_size, norm=opt.use_norm,
                                                               use_conv=opt.learned_attn, T=opt.T).to(self.device)
            self.g_reg_interval = opt.ppl_reg_every
            self.d_reg_interval = opt.r1_reg_every
            self.pl_mean = torch.tensor(0.0, device=self.device)

    def _get_current_device_for_loss_init(self):
        return torch.device(f'cuda:{self.opt.gpu_ids[0]}' if self.use_gpu() and self.opt.gpu_ids else 'cpu')

    def forward(self, data, mode, iter_count=0, arbitrary_input=False):
        content_image_orig = data['day'].to(self.device)
        night_image = data.get('night', content_image_orig).to(self.device)
        phi2_for_mixing = data.get('phi2', None)
        phi_for_interp = self.opt.phi.to(self.device, dtype=content_image_orig.dtype)

        w_day, w_night = get_day_night_weights(phi_for_interp, self.opt.daylight_curve_steepness)
        cos_phi = torch.cos(phi_for_interp)
        sin_phi = torch.sin(phi_for_interp)

        _, _, H, W = content_image_orig.shape
        light_map = create_light_direction_map(phi_for_interp, size=(H, W), num_channels=self.opt.light_map_channels,
                                               device=self.device, w_day=w_day)
        content_image_with_light = torch.cat([content_image_orig, light_map], dim=1)

        net_g_module = self.netG.module if isinstance(self.netG, torch.nn.DataParallel) else self.netG
        if hasattr(net_g_module, 'opt'): net_g_module.opt.phi = self.opt.phi

        if mode == 'inference':
            with torch.no_grad():
                fake_image, _ = net_g_module(content_image_with_light, w_day, w_night, cos_phi, sin_phi, phi2=None,
                                             arbitrary_input=arbitrary_input)
            return fake_image

        w_day_pix = w_day.view(-1, 1, 1, 1).expand_as(content_image_orig)
        style_image_target = w_day_pix * content_image_orig + (1.0 - w_day_pix) * night_image

        if mode == 'generator':
            g_loss_for_backward, g_losses_for_display, generated_images = self.compute_generator_loss(
                content_image_with_light, content_image_orig, style_image_target,
                night_image, w_day, w_night, cos_phi, sin_phi, iter_count, phi2_for_mixing,
                arbitrary_input=arbitrary_input
            )
            generated_images['style_target_interpolated'] = style_image_target
            generated_images['light_map_viz'] = light_map
            return g_loss_for_backward, g_losses_for_display, generated_images
        elif mode == 'discriminator':
            d1_losses, d2_losses = self.compute_discriminator_loss(
                content_image_with_light, content_image_orig, style_image_target,
                w_day, w_night, cos_phi, sin_phi, iter_count, phi2_for_mixing, arbitrary_input=arbitrary_input
            )
            return d1_losses, d2_losses
        else:
            raise ValueError(f"|mode| is invalid: {mode}")

    def g_main_loss(self, content_with_light, content_orig, style_target, night_image, w_day, w_night, cos_phi, sin_phi,
                    phi2_for_mixing, arbitrary_input, iter_count):
        G_losses = {}
        fake_image, identity_images_dict = self.netG(content_with_light, w_day, w_night, cos_phi, sin_phi,
                                                     phi2=phi2_for_mixing, arbitrary_input=arbitrary_input)
        if fake_image is None: raise RuntimeError("Generator output is None.")

        warmup_factor = min(1.0, iter_count / self.opt.night_loss_warmup_iters)
        G_losses['Night_Loss_Warmup'] = torch.tensor(warmup_factor, device=self.device)

        if self.opt.lambda_vgg_interp > 0 and self.criterionVGG is not None:
            with torch.no_grad():
                vgg_feats_day = self.criterionVGG.vgg(content_orig)
                soft_night_anchor_img = (content_orig * 0.1) - 0.85
                vgg_feats_soft_night = self.criterionVGG.vgg(soft_night_anchor_img)
                hard_night_anchor_img = torch.full_like(content_orig, -0.9, device=self.device)
                vgg_feats_hard_night = self.criterionVGG.vgg(hard_night_anchor_img)
                extreme_night_anchor_img = torch.full_like(content_orig, -1.0, device=self.device)
                vgg_feats_extreme_night = self.criterionVGG.vgg(extreme_night_anchor_img)

            vgg_feats_fake = self.criterionVGG.vgg(fake_image)
            vgg_interp_loss = 0

            phi_deg = self.opt.phi * (180.0 / math.pi)
            phi_deg = phi_deg.view(-1, 1, 1, 1)
            phi_deg_sym = torch.where(phi_deg > 180.0, 360.0 - phi_deg, phi_deg)

            alpha_p1 = (phi_deg_sym / 15.0).clamp(0.0, 1.0)
            alpha_p2 = ((phi_deg_sym - 15.0) / 30.0).clamp(0.0, 1.0)
            alpha_p3 = ((phi_deg_sym - 45.0) / 135.0).clamp(0.0, 1.0)

            for i in range(len(vgg_feats_fake)):
                interp_p1 = (1.0 - alpha_p1) * vgg_feats_extreme_night[i] + alpha_p1 * vgg_feats_hard_night[i]
                interp_p2 = (1.0 - alpha_p2) * vgg_feats_hard_night[i] + alpha_p2 * vgg_feats_soft_night[i]
                interp_p3 = (1.0 - alpha_p3) * vgg_feats_soft_night[i] + alpha_p3 * vgg_feats_day[i]

                mask_p1 = (phi_deg_sym <= 15.0).expand_as(vgg_feats_fake[i])
                mask_p2 = ((phi_deg_sym > 15.0) & (phi_deg_sym <= 45.0)).expand_as(vgg_feats_fake[i])

                target_feat = torch.where(mask_p1, interp_p1,
                                          torch.where(mask_p2, interp_p2,
                                                      interp_p3))

                vgg_interp_loss += self.criterionFeat(vgg_feats_fake[i], target_feat.detach())

            G_losses['VGG_Interp'] = vgg_interp_loss * self.opt.lambda_vgg_interp

        with torch.no_grad():
            hard_night_anchor = torch.full_like(content_orig, -1.0, device=self.device)
            soft_night_anchor = (content_orig * 0.1) - 0.85
            phi_deg_blend = self.opt.phi * (180.0 / math.pi)
            blend_weight = (F.relu(20.0 - phi_deg_blend) / 20.0).clamp(max=1.0).view(-1, 1, 1, 1)
            night_anchor_image = blend_weight * hard_night_anchor + (1.0 - blend_weight) * soft_night_anchor

        if self.netD2 is not None:
            pred_fake_d2_features, _ = self.discriminate(fake_image, content_orig, 'D2')
            G_losses['GAN_D2'] = self.criterionGAN(pred_fake_d2_features, True,
                                                   for_discriminator=False) * self.opt.lambda_G
            if not self.opt.no_ganFeat_loss and self.opt.lambda_feat > 0:
                real_target_for_d = w_day.view(-1, 1, 1, 1) * content_orig + w_night.view(-1, 1, 1,
                                                                                          1) * night_anchor_image
                with torch.no_grad():
                    pred_real_d2_features, _ = self.discriminate(real_target_for_d, content_orig, 'D2')
                G_losses['GAN_Feat_D2'] = self.feat_matching_loss(pred_fake_d2_features,
                                                                  pred_real_d2_features) * self.opt.lambda_feat

        if self.opt.lambda_identity > 0:
            i_cc = identity_images_dict.get('identity_content')
            if i_cc is not None:
                loss_identity = self.criterionIdentity(i_cc, content_orig)
                G_losses['Identity'] = (loss_identity * w_day).mean() * self.opt.lambda_identity

        if warmup_factor > 0:
            if self.opt.lambda_style_recon > 0:
                i_nn = identity_images_dict.get('identity_night')
                if i_nn is not None:
                    loss_style_recon = self.criterionIdentity(i_nn, night_anchor_image)
                    G_losses['Style_Recon'] = (
                                                      loss_style_recon * w_night).mean() * self.opt.lambda_style_recon * warmup_factor

            if self.opt.lambda_laplacian > 0 and hasattr(self, 'criterionLaplacian'):
                loss_lap = self.criterionLaplacian(fake_image, content_orig)
                G_losses['Laplacian'] = (loss_lap * w_night).mean() * self.opt.lambda_laplacian * warmup_factor

            if self.opt.lambda_tv > 0 and hasattr(self, 'criterionTV'):
                loss_tv = self.criterionTV(fake_image)
                G_losses['TV'] = (loss_tv * w_night).mean() * self.opt.lambda_tv * warmup_factor

            if self.opt.use_thresholded_blackout and hasattr(self, 'criterionThresholdedBlackout'):
                loss_thresh_blackout = self.criterionThresholdedBlackout(fake_image)
                G_losses['Thresh_Blackout'] = (
                                                      loss_thresh_blackout * w_night).mean() * self.opt.lambda_thresholded_blackout * warmup_factor

        if hasattr(self.opt, 'lambda_z_reg') and self.opt.lambda_z_reg > 0:
            netG_module = self.netG.module if isinstance(self.netG, torch.nn.DataParallel) else self.netG
            if hasattr(netG_module, 'z_day') and hasattr(netG_module, 'z_night'):
                z_day, z_night = netG_module.z_day, netG_module.z_night
                G_losses['Z_Reg'] = self.criterionFeat(z_day, z_night) * self.opt.lambda_z_reg

        total_loss = sum(l for l in G_losses.values() if l is not None and isinstance(l, torch.Tensor))
        return total_loss, G_losses, fake_image, identity_images_dict

    def compute_generator_loss(self, content_with_light, content_orig, style_target, night_image, w_day, w_night,
                               cos_phi, sin_phi, iter_count,
                               phi2_for_mixing, arbitrary_input=False):
        if self.netD2:
            for p in self.netD2.parameters(): p.requires_grad = False

        g_loss_main, g_losses_for_display, fake_image_1, identity_images_dict = self.g_main_loss(
            content_with_light, content_orig, style_target, night_image, w_day, w_night, cos_phi, sin_phi,
            phi2_for_mixing, arbitrary_input=arbitrary_input, iter_count=iter_count
        )

        if self.opt.lambda_continuity > 0:
            netG_module = self.netG.module if isinstance(self.netG, torch.nn.DataParallel) else self.netG

            phi_1 = self.opt.phi.clone()
            delta_phi = (torch.rand_like(phi_1) * 0.2) - 0.1
            phi_2 = (phi_1 + delta_phi)

            w_day_2, w_night_2 = get_day_night_weights(phi_2, self.opt.daylight_curve_steepness)
            cos_phi_2 = torch.cos(phi_2)
            sin_phi_2 = torch.sin(phi_2)

            _, _, H, W = content_orig.shape
            light_map_2 = create_light_direction_map(phi_2, (H, W), self.opt.light_map_channels, self.device, w_day_2)
            content_with_light_2 = torch.cat([content_orig, light_map_2], dim=1)

            with torch.no_grad():
                fake_image_2, _ = netG_module(content_with_light_2, w_day_2, w_night_2, cos_phi_2, sin_phi_2, phi2=None,
                                              arbitrary_input=arbitrary_input)

            loss_l1 = self.criterionIdentity(fake_image_1, fake_image_2.detach())
            loss_vgg = self.criterionVGG(fake_image_1,
                                         fake_image_2.detach()) if self.criterionVGG is not None else torch.tensor(0.0,
                                                                                                                   device=self.device)

            continuity_loss = loss_l1 + loss_vgg

            g_losses_for_display["Continuity"] = continuity_loss.clone().detach()
            g_loss_main += continuity_loss * self.opt.lambda_continuity

        if self.opt.lambda_ppl > 0 and (iter_count + 1) % self.g_reg_interval == 0:
            g_losses_for_display["PPL"] = torch.tensor(0.0, device=self.device)
        else:
            g_losses_for_display["PPL"] = torch.tensor(0.0, device=self.device)

        g_losses_for_display['G_total'] = g_loss_main.clone().detach()
        return g_loss_main, g_losses_for_display, {**identity_images_dict, 'synthesized_image': fake_image_1}

    def d_main_loss(self, content_with_light, content_orig, style_target, w_day, w_night, cos_phi, sin_phi,
                    phi2_for_mixing, arbitrary_input=False):
        if not self.netD2: return torch.tensor(0.0, device=self.device), {}
        for p in self.netD2.parameters(): p.requires_grad = True
        D2_losses = {}
        with torch.no_grad():
            fake_image, _ = self.netG(content_with_light, w_day, w_night, cos_phi, sin_phi, phi2=phi2_for_mixing,
                                      arbitrary_input=arbitrary_input)
            fake_image = fake_image.detach()
        pred_fake_d2_full, _ = self.discriminate(fake_image, content_orig, 'D2')
        D2_losses['D2_Fake'] = self.criterionGAN(pred_fake_d2_full, False, for_discriminator=True)
        pred_real_d2_full, _ = self.discriminate(style_target, content_orig, 'D2')
        D2_losses['D2_real'] = self.criterionGAN(pred_real_d2_full, True, for_discriminator=True)
        return sum(D2_losses.values()), D2_losses

    def compute_discriminator_loss(self, content_with_light, content_orig, style_target, w_day, w_night, cos_phi,
                                   sin_phi, iter_count, phi2_for_mixing, arbitrary_input=False):
        if not self.netD2: return {}, {}
        d_loss_main, d2_losses = self.d_main_loss(content_with_light, content_orig, style_target, w_day, w_night,
                                                  cos_phi, sin_phi, phi2_for_mixing, arbitrary_input=arbitrary_input)
        if self.opt.lambda_r1 > 0 and (iter_count + 1) % self.d_reg_interval == 0:
            r1_loss = self.d_r1_regularize(style_target.clone(), content_orig)
            d2_losses['D2_R1'] = r1_loss.clone().detach()
            weighted_r1_loss = self.opt.lambda_r1 / 2 * r1_loss * self.d_reg_interval
            d_loss_main += weighted_r1_loss
        else:
            d2_losses['D2_R1'] = torch.tensor(0.0, device=self.device)
        return {}, d2_losses

    def d_r1_regularize(self, style_target, content_orig):
        if not self.netD2: return torch.tensor(0.0, device=self.device)
        style_target.requires_grad = True
        _, pred_real_d2_logit = self.discriminate(style_target, content_orig, 'D2')
        r1_loss = self.r1_penalty(pred_real_d2_logit, style_target)
        return r1_loss

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        D_params = list(self.netD2.parameters()) if self.netD2 else []
        optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, opt.beta2)) if D_params else None
        return optimizer_G, optimizer_D, optimizer_D

    def r1_penalty(self, real_pred, real_img):
        grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True)[
            0]
        return grad_real.pow(2).reshape(real_img.shape[0], -1).sum(1).mean()

    def discriminate(self, image_to_judge, condition, type):
        target_D = self.netD2
        if 'swin' in self.opt.netD.lower():
            return target_D(image_to_judge, condition)
        else:
            d_input = torch.cat((image_to_judge, condition), dim=1)
            discriminator_out = target_D(d_input)
            if isinstance(discriminator_out, list):
                logit = discriminator_out[-1][-1] if isinstance(discriminator_out[-1], list) else discriminator_out[-1]
                return discriminator_out, logit
            else:
                return [discriminator_out], discriminator_out

    def feat_matching_loss(self, pred_fake_scales, pred_real_scales):
        loss = 0.0
        for fake_feat_scale, real_feat_scale in zip(pred_fake_scales, pred_real_scales):
            if isinstance(fake_feat_scale, list):
                for f, r in zip(fake_feat_scale[:-1], real_feat_scale[:-1]): loss += self.criterionFeat(f, r.detach())
            else:
                loss += self.criterionFeat(fake_feat_scale, real_feat_scale.detach())
        return loss

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if self.netD2: util.save_network(self.netD2, 'D2', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD2 = networks.define_D(opt) if opt.isTrain else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain: netD2 = util.load_network(netD2, 'D2', opt.which_epoch, opt)
        return netG, None, netD2, None

    def calculate_patch_vgg_loss(self, fake, real):
        if not (hasattr(self.opt,
                        'use_patch_vgg_loss') and self.opt.use_patch_vgg_loss and self.opt.lambda_patch_vgg > 0 and self.criterionVGG is not None): return torch.tensor(
            0.0, device=fake.device)
        patch_size, num_patches = self.opt.patch_vgg_size, self.opt.patch_vgg_num_patches
        B, C, H, W = fake.shape
        if patch_size > H or patch_size > W: return torch.tensor(0.0, device=fake.device)
        rand_h = torch.randint(0, H - patch_size + 1, (num_patches,), device=fake.device)
        rand_w = torch.randint(0, W - patch_size + 1, (num_patches,), device=fake.device)
        fake_patches, real_patches = [], []
        for i in range(num_patches):
            for b in range(B):
                fake_patches.append(
                    fake[b:b + 1, :, rand_h[i]:rand_h[i] + patch_size, rand_w[i]:rand_w[i] + patch_size])
                real_patches.append(
                    real[b:b + 1, :, rand_h[i]:rand_h[i] + patch_size, rand_w[i]:rand_w[i] + patch_size])
        if not fake_patches: return torch.tensor(0.0, device=fake.device)
        return self.criterionVGG(torch.cat(fake_patches, dim=0), torch.cat(real_patches, dim=0).detach())

    def calculate_style_loss(self, fake, target):
        if self.criterionVGG is None: return torch.tensor(0.0, device=fake.device)
        fake_vgg_feats = self.criterionVGG.vgg(fake)
        target_vgg_feats = self.criterionVGG.vgg(target)
        style_loss = 0.0
        for f_f, f_t in zip(fake_vgg_feats, target_vgg_feats):
            style_loss += self.criterionFeat(self.gram_matrix(f_f), self.gram_matrix(f_t).detach())
        return style_loss

    def gram_matrix(self, input_tensor):
        b, c, h, w = input_tensor.size()
        features = input_tensor.view(b, c, h * w)
        features_t = features.transpose(1, 2)
        G = features.bmm(features_t)
        return G.div(c * h * w)