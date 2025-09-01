# CR-Net: A Continuous Rendering Network for Enhancing Processing in Low-Light Environments

<p align="center">
    📄 <a href="link-to-your-paper"><b>Paper</b></a>&nbsp;&nbsp; | &nbsp;&nbsp;
    💻 <a href="https://github.com/val-utehy/CR-Net"><b>Source Code</b></a>&nbsp;&nbsp; | &nbsp;&nbsp;
    🤗 <a href="https://huggingface.co/datasets/datnguyentien204/CR-Net"><b>Hugging Face</b></a>
</p>

<p align="center">
    <img src="preview/structures.jpg" width="800"/>
<p>

<p align="center">
    <em>Architecture of the CR-Net model.</em>
<p>

## Introduction

**CR-Net** is a model enhance the quality of images and videos captured under low-light conditions. 
By learning a continuous rendering process, CR-Net effectively improves brightness, producing natural and sharp results even in challenging dark environments. 
To learn more about CR-Net, feel free to read our documentation [English](../README.md) | [Tiếng Việt](preview/README-vi.md) | [中文](preview/README-zh.md).


### Key Features

*   **Low-light image/video enhancement:** Significantly improves brightness and contrast for images and videos captured in dim lighting.
*   **Continuous rendering network:** Employs a novel architecture to deliver smoother and more natural results compared to traditional methods.
*   **Flexible applications:** Supports both video processing and directories containing multiple still images.

## Demo

![CR-Net Demo](preview/video_demo.gif)

## Installation and Requirements

To run this model, you need the proper environment. We recommend the following versions:

*   **Python:** `Python >= 3.10` (Recommended `Python 3.10`)
*   **PyTorch:** `PyTorch >= 1.12` (Recommended `PyTorch 2.1.2`)

**Step 1: Clone the repository**

```shell
  git clone https://github.com/val-utehy/CR-Net.git
  cd CR-Net
```
**Step 2: Install dependencies**

```shell
  pip install -r requirements.txt
```

> [!NOTE]
> Make sure you have installed the compatible versions of **torch** and **torchvision** with your **CUDA driver** to leverage GPU.
## Usage Guide

### 1. Model Training

Training file will be updated soon!

[//]: # (To train the CR-Net model on your own dataset, follow these steps:)

[//]: # ()
[//]: # (**a. Configure the training script file:**)

[//]: # ()
[//]: # (Open and edit the file `train_scripts/ast_n2h.sh`. In this file, you need to specify important paths such as the dataset path and the checkpoint saving directory.)

[//]: # ()
[//]: # (**b. Run the training script:**)

[//]: # ()
[//]: # (After finishing the configuration, navigate to the project’s root directory and execute the following command:)

[//]: # ()
[//]: # (```shell)

[//]: # (    bash train_scripts/ast_n2h_dat.sh)

[//]: # (```)
### 2. Testing and Inference

**a. Video Processing:**

#### 1. Configure the script file:
Open and edit the file `test_scripts/ast_inference_video_dat.sh`. Here, you need to provide the path to the trained checkpoint and the input/output video paths.

#### 2. Run the video processing script:
After completing the configuration, navigate to the project’s root directory and execute the following command:

```shell
  bash test_scripts/ast_inference_video.sh
```

**b. Image Directory Processing:**
#### 1. Configure the script file:
Open and edit the file `test_scripts/ast_n2h_dat.sh`. Here, you need to provide the path to the trained checkpoint and the input/output image directory paths.

#### 2. Run the image directory processing script:
After completing the configuration, navigate to the project’s root directory and execute the following command:

```shell
  bash test_scripts/ast_n2h.sh
``` 

## Citation


[//]: # (```bibtex)

[//]: # (@article{crnet2025,)

[//]: # (    title={CR-Net: A Continuous Rendering Network for Improving Robustness to Low-illumination},)

[//]: # (    author={},)

[//]: # (    journal={},)

[//]: # (    year={2025})

[//]: # (})

[//]: # (```)
## References

1. https://github.com/EndlessSora/TSIT

2. https://github.com/astra-vision/CoMoGAN

3. https://github.com/AlienZhang1996/S2WAT


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.