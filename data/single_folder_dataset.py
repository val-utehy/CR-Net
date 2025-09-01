# --- START OF FILE data/single_folder_dataset.py ---
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image


class SingleFolderDataset(BaseDataset):
    """
    A dataset class for loading images from a single folder.
    Used for testing where only content images are needed.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # is_train sẽ là False khi chạy test.py
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains images')
        # Khi test, chúng ta thường không muốn crop ngẫu nhiên,
        # nên preprocess_mode='resize' hoặc 'scale_width' là phù hợp.
        # Hoặc có thể giữ 'resize_and_crop' và dùng crop_pos=(0,0)
        parser.set_defaults(preprocess_mode='resize_and_crop', load_size=256, crop_size=256, no_flip=True)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.image_paths = sorted(make_dataset(opt.image_dir, recursive=True))
        # <<< THAY ĐỔI: Xóa dòng self.transform ở đây >>>

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = Image.open(path).convert('RGB')

        # <<< THAY ĐỔI: Tạo params và transform cho mỗi ảnh >>>
        # Khi test, chúng ta không muốn augmentation ngẫu nhiên.
        # Đặt crop_pos=(0,0) và flip=False để đảm bảo tính nhất quán.
        params = get_params(self.opt, img.size)
        if not self.opt.isTrain:
            params['crop_pos'] = (0, 0)
            params['flip'] = False

        transform = get_transform(self.opt, params, normalize=True)
        img_tensor = transform(img)
        # <<< KẾT THÚC THAY ĐỔI >>>

        # Trả về một dict tương thích với những gì model mong đợi ở bước test
        # 'day' sẽ được dùng làm content_image.
        return {'day': img_tensor, 'cpath': path}

    def __len__(self):
        return len(self.image_paths)