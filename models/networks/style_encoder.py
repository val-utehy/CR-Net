import torch
import torch.nn as nn
from models.networks.base_network import BaseNetwork
import functools



class SimpleStyleEncoder(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 4
        padw = 1
        use_bias_for_conv = True
        norm_layers_list = []
        if opt.norm_SE == 'instance':
            norm_layer_class = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
            use_bias_for_conv = True
        elif opt.norm_SE == 'batch':
            norm_layer_class = functools.partial(nn.BatchNorm2d, affine=True)
            use_bias_for_conv = False
        elif opt.norm_SE == 'none' or opt.norm_SE is None:
            norm_layer_class = None
        else:
            raise NotImplementedError(f"Normalization layer {opt.norm_SE} is not implemented for SimpleStyleEncoder")

        sequence = [
            nn.Conv2d(opt.style_enc_nc, opt.style_enc_nef, kernel_size=kw, stride=2, padding=padw,
                      bias=use_bias_for_conv),
        ]
        if norm_layer_class: sequence.append(norm_layer_class(opt.style_enc_nef))
        sequence.append(nn.ReLU(True))

        nf_mult = 1
        for n in range(1, opt.style_enc_n_downsampling):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence.append(
                nn.Conv2d(opt.style_enc_nef * nf_mult_prev, opt.style_enc_nef * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias_for_conv)
            )
            if norm_layer_class: sequence.append(norm_layer_class(opt.style_enc_nef * nf_mult))
            sequence.append(nn.ReLU(True))

        final_channels = opt.style_enc_nef * nf_mult

        sequence += [
            nn.Conv2d(final_channels, final_channels, kernel_size=3, stride=1, padding=1, bias=use_bias_for_conv),
        ]
        if norm_layer_class: sequence.append(norm_layer_class(final_channels))
        sequence.append(nn.ReLU(True))
        sequence += [nn.AdaptiveAvgPool2d(1)]  # Output: [B, final_channels, 1, 1]

        self.cnn_body = nn.Sequential(*sequence)
        self.fc_out = nn.Linear(final_channels, opt.style_enc_latent_dim)  # opt.style_enc_latent_dim

    def forward(self, input_image):
        features = self.cnn_body(input_image)
        features_flat = features.view(features.size(0), -1)  # Flatten
        latent_code = self.fc_out(features_flat)
        return latent_code