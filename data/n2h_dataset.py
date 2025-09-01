# --- START OF FILE data/n2h_dataset.py (Sửa lỗi AttributeError) ---
import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from PIL import Image
import random
from data.base_dataset import get_params, get_transform


class N2HDataset(Pix2pixDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        A_paths and B_paths are defined here, and we call the initialize
        method of the parent class (Pix2pixDataset) to set up the rest.
        """
        # Gọi __init__ của lớp cha gần nhất (Pix2pixDataset)
        # Pix2pixDataset không có __init__, nên nó sẽ gọi BaseDataset.__init__(self, opt)
        # Điều này là đúng với bản sửa lỗi trước của chúng ta.
        super().__init__(opt)

        # Gọi hàm initialize của lớp cha để thiết lập self.label_paths, self.image_paths, và self.dataset_size
        self.initialize(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=286)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        croot = opt.croot
        sroot = opt.sroot

        # Logic này giả định cấu trúc thư mục là croot/trainA, sroot/trainB
        c_image_dir = os.path.join(croot, opt.phase + 'A')
        s_image_dir = os.path.join(sroot, opt.phase + 'B')

        if not os.path.isdir(c_image_dir):
            raise FileNotFoundError(f"Content directory not found: {c_image_dir}")
        if not os.path.isdir(s_image_dir):
            raise FileNotFoundError(f"Style directory not found: {s_image_dir}")

        c_image_paths = sorted(make_dataset(c_image_dir, recursive=True))
        s_image_paths = sorted(make_dataset(s_image_dir, recursive=True))

        if opt.phase == 'train' and len(c_image_paths) > 0 and len(s_image_paths) > 0:
            if len(c_image_paths) > len(s_image_paths):
                s_image_paths = s_image_paths * (len(c_image_paths) // len(s_image_paths) + 1)
            elif len(s_image_paths) > len(c_image_paths):
                c_image_paths = c_image_paths * (len(s_image_paths) // len(c_image_paths) + 1)

        instance_paths = []

        return c_image_paths, s_image_paths, instance_paths

    def __getitem__(self, index):
        # Lấy ảnh Day (ảnh A - content)
        # self.label_paths được gán bằng c_image_paths trong Pix2pixDataset.initialize()
        day_path = self.label_paths[index % len(self.label_paths)]

        # Lấy ảnh Night (ảnh B - style) ngẫu nhiên
        # self.image_paths được gán bằng s_image_paths trong Pix2pixDataset.initialize()
        night_path = self.image_paths[random.randint(0, len(self.image_paths) - 1)]

        day_img = Image.open(day_path).convert('RGB')
        night_img = Image.open(night_path).convert('RGB')

        params = get_params(self.opt, day_img.size)
        transform = get_transform(self.opt, params)

        day_tensor = transform(day_img)
        night_tensor = transform(night_img)

        return {'day': day_tensor, 'night': night_tensor, 'cpath': day_path, 'spath_night': night_path}

    def paths_match(self, path1, path2):
        return True