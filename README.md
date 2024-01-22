# AgeGAN: Age Progression/Regression with GANs

## Overview

AgeGAN is a deep learning model that leverages Generative Adversarial Networks (GANs) for age progression and regression tasks on facial images. This repository includes a pretrained model for immediate inference on your images and guidelines for training your own model using either the CACD or UTK faces datasets.

## Getting Started

### Pretrained Model Inference

To apply the pretrained model to your images, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/AgeGAN.git
    cd AgeGAN
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run inference on your images:
    ```bash
    python infer.py --image_dir 'path/to/your/image/directory'
    ```

### Training Your Own Model

If you intend to train your own AgeGAN model, adhere to these steps:

#### Dataset Preparation

##### CACD Dataset
1. Download the CACD dataset.
2. Preprocess the dataset using the provided script:
    ```bash
    python preprocessing/preprocess_cacd.py --image_dir '/path/to/cacd/images' --metadata '/path/to/the/cacd/metadata/file' --output_dir 'path/to/save/processed/data'
    ```

##### UTK Faces Dataset
1. Download the UTK faces dataset.
2. Preprocess the dataset using the provided script:
    ```bash
    python preprocessing/preprocess_utk.py --data_dir '/path/to/utk/faces/images' --output_dir 'path/to/save/processed/data'
    ```

#### Model Training

1. Modify the configuration file (`configs/aging_gan.yaml`) to point to the processed dataset and adjust hyperparameters if needed.

2. Start the training:
    ```bash
    python main.py
    ```

#### Tensorboard Monitoring

Monitor the training progress using Tensorboard:

```bash
tensorboard --logdir=lightning_logs --bind_all
```

Visit `http://localhost:6006` in your web browser to visualize losses and generated images.

## Datasets

Download links for datasets:

- UTK Face dataset (Aligned & Cropped Faces): [UTKFace](https://susanqq.github.io/UTKFace/)
- CACD dataset: [CACD](https://bcsiriuschen.github.io/CARC/) (Note: Metadata only can be downloaded separately [here](https://bcsiriuschen.github.io/CARC/))