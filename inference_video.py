import os
import torch
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
import pickle

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from options.base_options import BaseOptions
from models.pix2pix_model import Pix2PixModel
from util import util
import torchvision.transforms as transforms
from PIL import Image

def load_options_from_checkpoint(opt):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    opt_file_path = os.path.join(expr_dir, 'opt.pkl')
    if not os.path.exists(opt_file_path):
        opt_file_path = os.path.join(expr_dir, 'opt.txt')
        if not os.path.exists(opt_file_path):
            raise FileNotFoundError(f"Không tìm thấy file option (opt.pkl hoặc opt.txt) trong {expr_dir}")
    print(f"Đang tải options từ: {opt_file_path}")
    try:
        with open(opt_file_path, 'rb') as f:
            loaded_opt = pickle.load(f)
    except Exception:
        loaded_opt = BaseOptions().load_options(opt)
    cmd_opt_vars = vars(opt)
    loaded_opt_vars = vars(loaded_opt)
    loaded_opt_vars.update(cmd_opt_vars)
    final_opt = argparse.Namespace(**loaded_opt_vars)
    final_opt.isTrain = False
    final_opt.phase = 'test'
    return final_opt

def get_safe_resolution(width, height, max_height=720):
    if height <= max_height and width % 16 == 0 and height % 16 == 0:
        return width, height
    ratio = max_height / height
    new_width = int(width * ratio)
    new_height = max_height
    if new_width % 16 != 0:
        new_width = new_width - (new_width % 16)
    if new_height % 16 != 0:
        new_height = new_height - (new_height % 16)
    return new_width, new_height

def preprocess_frame(frame_pil):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(frame_pil).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='Video inference script for Day-to-Night model.')
    parser.add_argument('--name', type=str, required=True, help='Tên của experiment đã train.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Thư mục chứa các checkpoints.')
    parser.add_argument('--which_epoch', type=str, default='latest', help='Epoch muốn sử dụng để test.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs. Dùng -1 cho CPU.')
    parser.add_argument('--video_path', type=str, required=True, help='Đường dẫn đến video đầu vào.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Thư mục gốc để lưu video kết quả.')
    parser.add_argument('--no_audio', action='store_true', help='Không giữ lại âm thanh từ video gốc.')
    parser.add_argument('--max_resolution_height', type=int, default=720, help='Chiều cao tối đa của video để xử lý.')
    parser.add_argument('--degree_step', type=float, default=5.0, help='Số độ sẽ tăng lên sau mỗi frame.')

    opt = parser.parse_args()
    opt = load_options_from_checkpoint(opt)
    opt.preprocess_mode = 'none'

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    model = Pix2PixModel(opt)
    model.eval()
    print(f"Model {opt.name} epoch {opt.which_epoch} loaded successfully.")

    video_in = cv2.VideoCapture(opt.video_path)
    if not video_in.isOpened():
        raise IOError(f"Không thể mở video: {opt.video_path}")

    fps = video_in.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc_width, proc_height = get_safe_resolution(original_width, original_height, opt.max_resolution_height)
    print(f"Original Video Info: {original_width}x{original_height}, {fps:.2f} FPS, {total_frames} frames.")
    if (proc_width, proc_height) != (original_width, original_height):
        print(f"Video will be resized to {proc_width}x{proc_height} for processing.")

    processed_frames_np = []

    for frame_idx in tqdm(range(total_frames), desc="Processing Video Frames"):
        ret, frame = video_in.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        input_tensor = preprocess_frame(frame_pil)
        data_i = {'day': input_tensor, 'cpath': ['dummy_path']}

        cycle_length = 360 / opt.degree_step
        half_cycle = int(cycle_length / 2)
        idx_in_cycle = frame_idx % int(cycle_length)
        if idx_in_cycle < half_cycle:
            current_degree = 180 + idx_in_cycle * opt.degree_step
        else:
            current_degree = (idx_in_cycle - half_cycle) * opt.degree_step

        phi_val = math.radians(current_degree)
        phi_tensor_1d = torch.tensor([phi_val], device=model.device)
        model.opt.phi = phi_tensor_1d

        with torch.no_grad():
            generated_tensor = model(data_i, mode='inference', arbitrary_input=True)
        generated_numpy = util.tensor2im(generated_tensor)[0]
        processed_frame_bgr = cv2.cvtColor(generated_numpy, cv2.COLOR_RGB2BGR)
        processed_frames_np.append(processed_frame_bgr)

    video_in.release()
    print("Finished processing all frames.")

    output_video_dir = os.path.join(opt.output_dir, opt.name, f'{opt.phase}_{opt.which_epoch}')
    os.makedirs(output_video_dir, exist_ok=True)
    video_basename = os.path.basename(opt.video_path)
    video_name, video_ext = os.path.splitext(video_basename)
    video_size, model_name=str(opt.max_resolution_height), str(opt.name)
    output_path = os.path.join(output_video_dir, f"{video_name}_cycle_by_degree{video_ext}_{video_size}_{model_name}.mp4")
    print(f"Saving output video to: {output_path}")

    temp_video_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(temp_video_path, fourcc, fps, (proc_width, proc_height))
    for frame in tqdm(processed_frames_np, desc="Writing video file"):
        video_out.write(frame)
    video_out.release()

    if not opt.no_audio and MOVIEPY_AVAILABLE:
        print("Adding audio from original video...")
        try:
            original_clip = VideoFileClip(opt.video_path)
            generated_clip = VideoFileClip(temp_video_path)
            final_clip = generated_clip.set_audio(original_clip.audio)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger='bar')
            original_clip.close()
            generated_clip.close()
            os.remove(temp_video_path)
            print("Audio added successfully.")
        except Exception as e:
            print(f"\nWarning: Could not add audio due to an error: {e}")
            print("The video without audio has been saved.")
            os.rename(temp_video_path, output_path)
    else:
        if not MOVIEPY_AVAILABLE and not opt.no_audio:
            print("\nWarning: 'moviepy' is not installed. Cannot process audio. Skipping.")
        os.rename(temp_video_path, output_path)
    print(f"Video processing complete. Output saved to {output_path}")

if __name__ == '__main__':
    main()
