import os
import ntpath
import time
from . import util
from . import html
import numpy as np
from PIL import Image as PILImage
import torch
from collections import OrderedDict

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def convert_map_to_numpy(self, data_map):
        if data_map is None or not isinstance(data_map, torch.Tensor):
            return None
        if data_map.dim() == 4:
            data_map = data_map[0]
        if data_map.size(0) > 1:
            data_map = data_map[0, :, :].unsqueeze(0)
        map_numpy = data_map.cpu().float().numpy()
        min_val, max_val = np.min(map_numpy), np.max(map_numpy)
        if max_val - min_val > 1e-6:
            map_numpy = (map_numpy - min_val) / (max_val - min_val)
        else:
            map_numpy = np.zeros_like(map_numpy)
        map_numpy = (map_numpy * 255.0).astype(np.uint8)
        if map_numpy.shape[0] == 1:
            map_numpy = np.transpose(map_numpy, (1, 2, 0))
            map_numpy = np.repeat(map_numpy, 3, axis=2)
        else:
            map_numpy = np.stack((map_numpy,) * 3, axis=-1)
        return map_numpy

    def display_current_results(self, visuals, epoch, step):
        visuals_np = OrderedDict()
        for label, image in visuals.items():
            if image is None:
                continue
            if 'light_map' in label:
                image_numpy = self.convert_map_to_numpy(image)
            elif 'input_label' in label:
                image_numpy = util.tensor2label(image, self.opt.label_nc, tile=False)
            else:
                image_numpy = util.tensor2im(image, tile=False)

            if image_numpy.ndim == 4:
                image_numpy = image_numpy[0]

            visuals_np[label] = image_numpy

        if self.tf_log:
            img_summaries = []
            for label, image_numpy in visuals_np.items():
                if image_numpy is None: continue
                try:
                    s = BytesIO()
                    pil_img = PILImage.fromarray(image_numpy)
                    pil_img.save(s, format="jpeg")
                    img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0],
                                                    width=image_numpy.shape[1])
                    img_summaries.append(self.tf.Summary.Value(tag=f'epoch_{epoch}/{label}', image=img_sum))
                except Exception as e:
                    print(f"Could not write image {label} to TF logs: {e}")

            if img_summaries:
                summary = self.tf.Summary(value=img_summaries)
                self.writer.add_summary(summary, step)

        if self.use_html:
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            webpage.add_header('Epoch [%d] Iteration [%d]' % (epoch, step))

            visuals_for_html = []
            labels_for_html = []

            standard_height = self.opt.crop_size

            for label, image_numpy in visuals_np.items():
                if image_numpy is None: continue

                pil_img = PILImage.fromarray(image_numpy)

                if pil_img.height != standard_height:
                    aspect_ratio = pil_img.width / pil_img.height
                    new_width = int(standard_height * aspect_ratio)
                    pil_img = pil_img.resize((new_width, standard_height), PILImage.LANCZOS)

                visuals_for_html.append(np.array(pil_img))
                labels_for_html.append(label)

            if not visuals_for_html:
                return

            try:
                concatenated_image = np.concatenate(visuals_for_html, axis=1)

                image_name = 'epoch%.3d_iter%.7d_combined.png' % (epoch, step)
                save_path = os.path.join(self.img_dir, image_name)
                util.save_image(concatenated_image, save_path)

                webpage.add_images([image_name], [' | '.join(labels_for_html)], [image_name],
                                   width=self.win_size * len(visuals_for_html))
                webpage.save()

            except ValueError as e:
                print(f"Error during HTML image concatenation for step {step}: {e}")
                print("Skipping HTML log for this step. Image shapes might be incompatible even after resizing.")

    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                if isinstance(value, torch.Tensor):
                    value_to_log = value.mean().float().item()
                elif isinstance(value, (float, int)):
                    value_to_log = float(value)
                else:
                    continue
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value_to_log)])
                self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v_orig in errors.items():
            v_to_print = v_orig
            if isinstance(v_orig, torch.Tensor):
                if v_orig.numel() > 0:
                    v_to_print = v_orig.mean().item()
                else:
                    v_to_print = 0.0
            elif not isinstance(v_orig, (float, int)):
                continue

            message += '%s: %.3f ' % (k, float(v_to_print))

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, webpage, visuals, image_path_list, alpha=1.0):
        visuals_np = OrderedDict()
        for label, image in visuals.items():
            if 'light_map' in label:
                visuals_np[label] = self.convert_map_to_numpy(image)
            else:
                visuals_np[label] = util.tensor2im(image)

        base_image_dir = webpage.get_image_dir()
        image_path_str = image_path_list[0] if isinstance(image_path_list, (list, tuple)) else image_path_list
        short_path = ntpath.basename(image_path_str)
        name_prefix = os.path.splitext(short_path)[0]

        current_alpha_float = alpha
        if isinstance(current_alpha_float, torch.Tensor):
            current_alpha_float = current_alpha_float.mean().item()
        elif not isinstance(current_alpha_float, (float, int)):
            try:
                current_alpha_float = float(current_alpha_float)
            except ValueError:
                current_alpha_float = 1.0

        alpha_folder_name = "alpha_{:.3f}".format(current_alpha_float).replace('.', '_')
        specific_alpha_image_dir = os.path.join(base_image_dir, alpha_folder_name)
        util.mkdirs(specific_alpha_image_dir)

        image_name_final = '%s.png' % (name_prefix)
        save_path = os.path.join(specific_alpha_image_dir, image_name_final)

        images_to_concatenate = []
        for label, image_numpy in visuals_np.items():
            img_to_add = image_numpy
            if image_numpy.ndim == 4 and image_numpy.shape[0] == 1:
                img_to_add = image_numpy.squeeze(0)
            elif image_numpy.ndim != 2 and image_numpy.ndim != 3:
                continue

            if img_to_add.ndim == 2:
                img_to_add = np.stack((img_to_add,) * 3, axis=-1)
            if img_to_add.ndim == 3 and img_to_add.shape[2] == 1:
                img_to_add = np.concatenate([img_to_add] * 3, axis=2)

            if img_to_add.shape[2] == 3:
                images_to_concatenate.append(img_to_add)

        if not images_to_concatenate:
            return

        try:
            image_concatenated_horizontally = np.concatenate(images_to_concatenate, axis=1)
            util.save_image(image_concatenated_horizontally, save_path, create_dir=True)
        except ValueError as e:
            print(f"Error concatenating images for {save_path}: {e}")
            print("Concatenated images list content (shapes):")
            for idx, vis_np_item in enumerate(images_to_concatenate):
                print(f"  Visual {idx}: shape {vis_np_item.shape if hasattr(vis_np_item, 'shape') else 'N/A'}")

        relative_image_path_for_html = os.path.join(alpha_folder_name, image_name_final)
        webpage.add_images([relative_image_path_for_html], [f"{name_prefix}_alpha_{current_alpha_float:.3f}"],
                           [relative_image_path_for_html], width=self.win_size * len(images_to_concatenate))