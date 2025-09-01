import os
import torch
import cv2
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import argparse
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util import util


def get_safe_resolution(width, height, max_height):
    if height <= max_height and width % 16 == 0 and height % 16 == 0:
        return width, height

    if height > max_height:
        ratio = max_height / float(height)
        new_width = int(width * ratio)
        new_height = max_height
    else:
        new_width = width
        new_height = height

    safe_width = new_width - (new_width % 16)
    safe_height = new_height - (new_height % 16)

    if safe_width == 0: safe_width = 16
    if safe_height == 0: safe_height = 16

    return safe_width, safe_height


class CustomTestOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--hours', type=float, nargs='+', required=True,
                            help='List of hours to apply (e.g., 8.5 13 18.75). Range: 0h-24h.')
        parser.add_argument('--add_timestamp', action='store_true',
                            help='If enabled, a timestamp (HH:MM) will be added at the top-left corner of the image.')
        parser.add_argument('--max_height', type=int, default=720,
                            help='Maximum height for processed images. If set, images will be resized. Example: 720.')
        parser.set_defaults(preprocess_mode='none')
        return parser


def main():
    opt = CustomTestOptions().parse()

    model = Pix2PixModel(opt)
    model.eval()
    print(f"Model {opt.name} epoch {opt.which_epoch} has been successfully loaded.")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    try:
        image_files = sorted(
            [f for f in os.listdir(opt.image_dir) if os.path.splitext(f)[1].lower() in image_extensions])
    except FileNotFoundError:
        raise IOError(f"Error: Cannot find input image directory: {opt.image_dir}")

    if not image_files:
        raise IOError(f"No image files found in directory: {opt.image_dir}")

    if opt.how_many < len(image_files):
        image_files = image_files[:int(opt.how_many)]
        print(f"Processing the first {len(image_files)} images according to --how_many parameter.")

    output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    total_iterations = len(image_files) * len(opt.hours)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    printed_resize_message = False

    with tqdm(total=total_iterations, desc="Processing images") as pbar:
        for image_name in image_files:
            image_path = os.path.join(opt.image_dir, image_name)

            try:
                input_image_pil = Image.open(image_path).convert('RGB')
                original_width, original_height = input_image_pil.size
                image_to_process = input_image_pil

                if opt.max_height is not None and opt.max_height > 0:
                    proc_width, proc_height = get_safe_resolution(original_width, original_height, opt.max_height)
                    if (proc_width, proc_height) != (original_width, original_height):
                        image_to_process = input_image_pil.resize((proc_width, proc_height), Image.Resampling.LANCZOS)
                        if not printed_resize_message:
                            print(
                                f"Note: Large images will be resized. Example: {original_width}x{original_height} -> {proc_width}x{proc_height}.")
                            printed_resize_message = True

                input_tensor = transform(image_to_process).unsqueeze(0)
                data_i = {'day': input_tensor, 'cpath': [image_path]}

                for hour in opt.hours:
                    angle_degrees = hour * 15.0
                    phi_val = math.radians(angle_degrees)
                    phi_tensor_1d = torch.tensor([phi_val], device=model.device)
                    model.opt.phi = phi_tensor_1d

                    with torch.no_grad():
                        generated_tensor = model(data_i, mode='inference', arbitrary_input=True)

                    generated_numpy_rgb = util.tensor2im(generated_tensor)[0]

                    if image_to_process.size != input_image_pil.size:
                        generated_pil = Image.fromarray(generated_numpy_rgb)
                        generated_pil_resized = generated_pil.resize(input_image_pil.size, Image.Resampling.LANCZOS)
                        generated_numpy_rgb = np.array(generated_pil_resized)

                    generated_bgr = cv2.cvtColor(generated_numpy_rgb, cv2.COLOR_RGB2BGR)

                    if opt.add_timestamp:
                        h = int(hour)
                        m = int((hour - h) * 60)
                        timestamp_text = f"{h:02d}:{m:02d}"

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 1
                        padding = 8

                        (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, font, font_scale,
                                                                              thickness)

                        margin = 10
                        rect_tl = (margin, margin)
                        rect_br = (margin + text_width + padding * 2, margin + text_height + padding * 2)

                        text_origin = (margin + padding, margin + text_height + padding)

                        cv2.rectangle(generated_bgr, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)

                        cv2.putText(generated_bgr, timestamp_text, text_origin, font, font_scale,
                                    (255, 255, 255), thickness, cv2.LINE_AA)
                    base_name, ext = os.path.splitext(image_name)
                    output_filename = f"{base_name}_hour_{hour}{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, generated_bgr)

                    pbar.update(1)

            except Exception as e:
                print(f"\nError while processing image {image_name}: {e}")
                if 'cuda' in str(e).lower():
                    torch.cuda.empty_cache()
                remaining_hours = len(opt.hours) - (pbar.n % len(opt.hours))
                if remaining_hours < len(opt.hours):
                    pbar.update(remaining_hours)

    print(f"\nProcessing completed. Results are saved in: {output_dir}")


if __name__ == '__main__':
    main()
