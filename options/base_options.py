import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='ast_s2wat_default_experiment',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--task', type=str, default='AST', help='task type: AST | SIS | MMIS')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix',
                            help='which model to use (pix2pix is the main model structure)')
        parser.add_argument('--norm_G', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization for G parts not using FADE/S2WAT norms')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization for D')
        parser.add_argument('--norm_S', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization for Stream (original TSIT content/style stream)')
        parser.add_argument('--norm_E', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization for E (TSIT VAE encoder)')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')


        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop',
                            help='scaling and cropping of images at load time.',
                            choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop",
                                     "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=286,
                            help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=3,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataset_mode', type=str, default='summer2winteryosemite')
        parser.add_argument('--croot', type=str, default='./datasets/summer2winter_yosemite/', help='content dataroot')
        parser.add_argument('--sroot', type=str, default='./datasets/summer2winter_yosemite/', help='style dataroot')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true',
                            help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='RafaelGenerator',
                            help='selects model to use for netG (tsit | pix2pixhd | RafaelGenerator)')
        parser.add_argument('--ngf', type=int, default=64,
                            help='# of gen filters in first conv layer (used by original TSIT, S2WAT uses embed_dim)')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')

        # <<< THAY ĐỔI: Đổi tên alpha thành phi
        parser.add_argument('--phi', type=float, default=0.0,
                            help='The parameter that controls the degree of stylization cyclically (angle in radians, 0 to 2*pi)')

        # for instance-wise features (original TSIT)
        parser.add_argument('--no_instance', action='store_true',
                            help='if specified, do *not* add instance map as input (for SIS task)')
        parser.add_argument('--use_vae', action='store_true',
                            help='enable training with an image encoder (TSIT original VAE).')

        self.initialized = True
        return parser

    def gather_options(self):
        """
        Gathers all options, initializes the parser, and parses arguments.
        This version is corrected to avoid UnboundLocalError.
        """
        # Always create a parser instance. The initialize() method will add arguments to it.
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # The initialize method adds all options and sets self.initialized = True
        parser = self.initialize(parser)

        # The rest of the original logic can now run safely
        opt, unknown = parser.parse_known_args()

        # Add model and dataset specific options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        # Re-parse to get all options, including model/dataset specific ones
        opt, _ = parser.parse_known_args()

        # Load options from file if requested
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        # Final parse
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>35}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt_path = file_name + '.pkl'
        if not os.path.exists(new_opt_path):
            print(f"Warning: Option file {new_opt_path} not found. Using command line options.")
            return opt
        new_opt = pickle.load(open(new_opt_path, 'rb'))
        return new_opt

    def parse(self, save=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.netG == 'RafaelGenerator':
            if hasattr(opt, 'rafael_img_size') and opt.rafael_img_size != opt.crop_size:
                print(f"Warning: crop_size ({opt.crop_size}) and rafael_img_size ({opt.rafael_img_size}) differ. "
                      f"Setting rafael_img_size to crop_size ({opt.crop_size}) for consistency during training.")
                opt.rafael_img_size = opt.crop_size

        if opt.use_vae:
            if not hasattr(opt, 'latent_dim'):
                print("Warning: --latent_dim not found, VAE might not be configured correctly. ")

        assert opt.task == 'AST' or opt.task == 'SIS' or opt.task == 'MMIS', \
            f'Task type should be: AST | SIS | MMIS, but got {opt.task}.'

        if opt.task == 'SIS':
            opt.semantic_nc = opt.label_nc + \
                              (1 if opt.contain_dontcare_label else 0) + \
                              (0 if opt.no_instance else 1)
            if opt.netG == 'RafaelGenerator':
                print(
                    "Warning: Using RafaelGenerator for SIS task. Ensure rafael encoder can handle semantic maps or a conversion layer exists.")
        else:
            opt.semantic_nc = 3

        self.print_options(opt)
        if opt.isTrain and save:
            self.save_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id_val = int(str_id)
            if id_val >= 0:
                opt.gpu_ids.append(id_val)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt