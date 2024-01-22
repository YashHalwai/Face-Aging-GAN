# AgeGAN: Age Progression/Regression with GANs

## Overview

AgeGAN is a deep learning model that utilizes Generative Adversarial Networks (GANs) for age progression and regression on facial images. This repository provides a pretrained model for immediate inference on your own images and comprehensive instructions for training your own model using the CACD or UTK Faces datasets.

## Getting Started

### Pretrained Model Inference

To test the pretrained model on your images, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AgeGAN.git
    cd AgeGAN
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run inference on your images:**
    ```bash
    python infer.py --image_dir 'path/to/your/image/directory'
    ```

### Training Your Own Model

If you wish to train your own AgeGAN model, follow these steps:

#### Dataset Preparation

##### CACD Dataset
1. **Download the CACD dataset.**
2. **Preprocess the dataset using the provided script:**
    ```bash
    python preprocessing/preprocess_cacd.py --image_dir '/path/to/cacd/images' --metadata '/path/to/the/cacd/metadata/file' --output_dir 'path/to/save/processed/data'
    ```

##### UTK Faces Dataset
1. **Download the UTK faces dataset.**
2. **Preprocess the dataset using the provided script:**
    ```bash
    python preprocessing/preprocess_utk.py --data_dir '/path/to/utk/faces/images' --output_dir 'path/to/save/processed/data'
    ```

#### Model Training

1. **Modify the configuration file (`configs/aging_gan.yaml`) to point to the processed dataset and adjust hyperparameters if needed.**

2. **Start the training:**
    ```bash
    python main.py
    ```

#### Tensorboard Monitoring

Monitor the training progress using Tensorboard:

```bash
tensorboard --logdir=lightning_logs --bind_all
```

Visit `http://localhost:6006` in your web browser to visualize losses and generated images.

### Download Datasets

- UTK Face dataset (Aligned & Cropped Faces): [Download Link](https://susanqq.github.io/UTKFace/)
- CACD dataset: [Download Link](https://bcsiriuschen.github.io/CARC/)

Ensure to comply with the dataset licenses and terms of use.

Feel free to reach out for any issues or improvements!