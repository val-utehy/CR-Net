import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import AdaptiveInstanceNorm
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import FADEResnetBlock as FADEResnetBlock
from models.networks.dstream import Stream as Dstream
from models.networks.architecture import FCMapping

class TSITGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralfadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks."
                                 "If 'most', also add one more upsampling + resnet layer at the end of the generator."
                                 "We only use 'more' as the default setting.")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.content_stream = Dstream(self.opt)
        self.fc_mapping = FCMapping(self.opt)
        self.block_config = [16, 16, 8, 4 ,2]
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        total_feat = 856
        growth_rate = 32
        self.fader_feats = nn.ModuleList()
        self.norms = nn.ModuleList()

        num_features = total_feat
        for i, num_layers in enumerate(self.block_config):
            out_feat = num_features - (num_layers * growth_rate)
            if out_feat < 0: out_feat = num_features * 2 - (num_layers * growth_rate)
            if out_feat == 0: out_feat = num_features
            fader_resnet_block = FADEResnetBlock(num_features, out_feat, opt)
            norm = AdaptiveInstanceNorm(num_features, opt)
            self.fader_feats.add_module("up_%d" % (i), fader_resnet_block)
            self.norms.add_module("adain_%d" % (i), norm)
            num_features = out_feat

        # last one more layer
        final = FADEResnetBlock(num_features, out_feat, opt)
        self.fader_feats.add_module("up_%d" % (i+1), final)
        fnorm = AdaptiveInstanceNorm(num_features, opt)
        self.norms.add_module("adain_%d" % (i + 1), fnorm)
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map (content) instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, total_feat, 3, padding=1)

        self.conv_img = nn.Conv2d(num_features, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt, num_blocks=5):

        sw = opt.crop_size // (2**(num_blocks+1))
        sh = round(sw / opt.aspect_ratio)

        return sw, sh


    def forward(self, input, real, z=None):
            content = input
            style =  real
            ft0, ft1, ft2, ft3, ft4, ft5 = self.content_stream(content)
            style_latent = self.fc_mapping(style)
            # style_latent = None

            # sample random noise
            x = torch.randn(content.size(0), 3, self.sw, self.sh, dtype=torch.float32, device=content.get_device())
            x = self.fc(x)

            # get alpha from options (should be random during training)
            alpha = self.opt.alpha

            x = self.norms.adain_0(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_0(x, ft5)
            x = self.up(x)

            x = self.norms.adain_1(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_1(x, ft4)
            x = self.up(x)

            x = self.norms.adain_2(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_2(x, ft3)
            x = self.up(x)

            x = self.norms.adain_3(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_3(x, ft2)
            x = self.up(x)

            x = self.norms.adain_4(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_4(x, ft1)
            x = self.up(x)

            x = self.norms.adain_5(x, style_latent, alpha=alpha)
            x = self.fader_feats.up_5(x, ft0)
            x = self.up(x)

            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = torch.tanh(x)
            return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = self.get_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
