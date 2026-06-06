# Generative AI — Assignment 03
**AI4009 | Spring 2026 | Batch 22F**

## Repository Structure

GenAI_Ass03/

q1-tackling-mode-collapse-in-gans-dcgan-vs-wgan.ipynb (Question 1: DCGAN vs WGAN-GP)

Q2_3339_3859.ipynb (Question 2: Pix2Pix)

Q3_3339_3859.ipynb (Question 3: CycleGAN)

app.py (CycleGAN Gradio Deployment)

## Question 1: Tackling Mode Collapse in GANs

### Objective
Compare DCGAN and WGAN-GP to reduce mode collapse and improve image diversity.

### Datasets
Pokemon Sprites

Anime Faces (64×64)

### Architectures

**DCGAN**
Input noise: 100-dim

Image size: 64×64

Generator: ConvTranspose → BatchNorm → ReLU → Tanh

Discriminator: Conv → LeakyReLU → Sigmoid

**WGAN-GP**
Critic replaces discriminator

Wasserstein loss with gradient penalty (λ=10)

5 critic updates per generator update

### Training Configuration

IMG_SIZE = 64

NOISE_DIM = 100

BATCH_SIZE = 64

LR = 0.0002

BETAS = (0.5, 0.999)

EPOCHS_DCGAN = 50

EPOCHS_WGAN = 60

LAMBDA_GP = 10

N_CRITIC = 5

SUBSET = 5000

### Techniques
Mixed precision training

Dual GPU (T4 × 2)

Checkpoint saving every 10 epochs

Dataset subset for faster training

### Results
Loss curves for generator and discriminator/critic

Generated image samples

Visual diversity comparison between DCGAN and WGAN-GP

---

## Question 2: Doodle-to-Real Image Translation using Pix2Pix

### Objective
Paired image-to-image translation:

Sketch → Real Image

Grayscale → Color Image

### Datasets
CUHK Face Sketch (CUFS)

Anime Sketch Colorization

### Architecture

**Generator (U-Net)**
Encoder-decoder with skip connections

Input: 256×256 sketch/grayscale image

Output: realistic/colored image

**Discriminator (PatchGAN)**
Patch-based classification

Classifies image patches as real or fake

### Training Configuration

IMG_SIZE = 256

BATCH_SIZE = 16–32

LR = 0.0002

BETAS = (0.5, 0.999)

### Loss Function

Total Loss = GAN Loss + λ × L1 Reconstruction Loss

### Techniques
Mixed precision training

Paired supervised learning

Checkpoint saving every 5–10 epochs

### Evaluation
SSIM

PSNR

Visual comparison (Input, Generated, Ground Truth)

---

## Question 3: Unpaired Image Translation using CycleGAN

### Objective
Unpaired image translation:

Sketch → Photo

Photo → Sketch

Cycle consistency for structure preservation

### Datasets
TU-Berlin Sketch

Sketchy Dataset

Google QuickDraw

### Architecture

**Generators**
G_AB: Sketch → Photo

G_BA: Photo → Sketch

6 ResNet blocks

Image size: 128×128

**Discriminators**
D_A: Sketch domain

D_B: Photo domain

PatchGAN based

### Training Configuration

IMG_SIZE = 128

BATCH_SIZE = 4–8

LR = 0.0002

BETAS = (0.5, 0.999)

N_RESBLOCKS = 6

### Loss Function

Total Loss = Adversarial Loss + Cycle Consistency Loss + Identity Loss

### Techniques
Mixed precision training

Dual GPU (T4 × 2)

Frequent checkpointing

Dataset subset for memory efficiency

### Evaluation
SSIM

PSNR

Visual translation and reconstruction results

---

## Gradio App Deployment

Real-time Sketch → Photo conversion using the trained CycleGAN model.

### Features
Upload a sketch image

Generate a realistic photo instantly

Simple Gradio interface

### Run

pip install gradio torch torchvision pillow

python app.py

Place `cyc_G_BA_ep50.pth` in the project directory before running.

---

## Environment

Platform: Kaggle

GPU: T4 × 2

Framework: PyTorch

Mixed Precision: torch.cuda.amp

App Framework: Gradio

### Dependencies

pip install torch torchvision gradio scikit-image pillow numpy matplotlib

---

## Notes

Models trained on Kaggle using dual T4 GPUs.

Dataset subsets were used when needed for memory efficiency.

Checkpoints saved every 5–10 epochs.

Mixed precision training used throughout for faster training and lower memory usage.

**AI4009 — Generative AI | FAST-NUCES**
