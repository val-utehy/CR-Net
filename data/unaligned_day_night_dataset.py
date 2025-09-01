import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random

class UnalignedDayNightDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders train/val containing day/night)')
        parser.set_defaults(preprocess_mode='resize_and_crop', load_size=286, crop_size=256)
        if not is_train:
            parser.set_defaults(no_flip=True)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self)
        self.opt = opt

        root = opt.dataroot
        phase = opt.phase

        self.dir_A = os.path.join(root, phase, 'day')
        self.dir_B = os.path.join(root, phase, 'night')

        self.A_paths = sorted(make_dataset(self.dir_A, recursive=True))
        self.B_paths = sorted(make_dataset(self.dir_B, recursive=True))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.A_size == 0 or self.B_size == 0:
            raise (RuntimeError(f"Found 0 images in one of the data directories: {self.dir_A} or {self.dir_B}"))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        params = get_params(self.opt, A_img.size)
        transform = get_transform(self.opt, params)

        A = transform(A_img)
        B = transform(B_img)

        return {'day': A, 'night': B, 'cpath': A_path, 'spath_night': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)