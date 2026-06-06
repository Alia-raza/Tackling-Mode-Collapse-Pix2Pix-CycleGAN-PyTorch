# 🤖 Generative AI — Assignment 03
### AI4009 | Spring 2026 | Batch 22F

**Group Members:**
| Roll Number | Name |
|-------------|------|
| 22F-3859 | *(Member 1)* |
| 22F-3339 | *(Member 2)* |

---

## 📁 Repository Structure

```
GenAI_Ass03/
│
├── q1-tackling-mode-collapse-in-gans-dcgan-vs-wgan.ipynb   # Question 1: DCGAN vs WGAN-GP
├── Q2_3339_3859.ipynb                                        # Question 2: Pix2Pix
├── Q3_3339_3859.ipynb                                        # Question 3: CycleGAN
└── app.py                                                    # Gradio App (CycleGAN Deployment)
```

---

## 🧠 Question 1: Tackling Mode Collapse in GANs (DCGAN vs WGAN-GP)

### 🎯 Objective
Design and implement a GAN system that addresses mode collapse using:
- **Baseline**: Deep Convolutional GAN (DCGAN)
- **Improved**: Wasserstein GAN with Gradient Penalty (WGAN-GP)

### 📦 Datasets
| Dataset | Link |
|---------|------|
| 🎮 Pokemon Sprites | [Kaggle](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites) |
| 🎌 Anime Faces (64×64) | [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) |

### 🏗️ Architecture

#### DCGAN (Baseline)
- **Input Noise Vector (z):** 100-dimensional
- **Image Size:** 64 × 64
- **Generator:** Transposed Conv → BatchNorm → ReLU → Tanh
- **Discriminator:** Conv → LeakyReLU → Sigmoid

#### WGAN-GP (Advanced)
- **Critic** replaces Discriminator (no sigmoid)
- **Loss:** Wasserstein Loss + Gradient Penalty (λ = 10)
- **Critic updates per Generator update:** 5

### ⚙️ Training Config
```python
IMG_SIZE    = 64
NOISE_DIM   = 100
BATCH_SIZE  = 64
LR          = 0.0002
BETAS       = (0.5, 0.999)
EPOCHS_DCGAN = 50
EPOCHS_WGAN  = 60
LAMBDA_GP   = 10
N_CRITIC    = 5
SUBSET      = 5000
```

### 🔧 Techniques Used
- ✅ Mixed Precision Training (`torch.cuda.amp`)
- ✅ Dual GPU (Kaggle T4 × 2)
- ✅ Checkpoint saving every 10 epochs
- ✅ Dataset subset for faster training

### 📊 Results
- Generator and Discriminator/Critic loss plots across epochs
- 5–10 generated image samples per model
- Visual comparison: DCGAN vs WGAN-GP diversity

---

## 🎨 Question 2: Doodle-to-Real Image Translation using Pix2Pix

### 🎯 Objective
Paired image-to-image translation using Conditional GAN (Pix2Pix):
- **Sketch → Realistic Image**
- **Grayscale → Colored Image**

### 📦 Datasets
| Dataset | Link |
|---------|------|
| 👤 CUHK Face Sketch (CUFS) | [Kaggle](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs) |
| 🖌️ Anime Sketch Colorization | [Kaggle](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) |

### 🏗️ Architecture

#### Generator — U-Net
- Encoder-Decoder with skip connections
- Input: Sketch / Grayscale (256 × 256)
- Output: Realistic / Colored image

#### Discriminator — PatchGAN
- Patch-based classification (16 × 16 patches)
- Classifies whether each patch is real or fake

### ⚙️ Training Config
```python
IMG_SIZE   = 256
BATCH_SIZE = 16–32
LR         = 0.0002
BETAS      = (0.5, 0.999)
```

### 📉 Loss Functions
```
Total Loss = Adversarial Loss (GAN) + λ × L1 Loss (Reconstruction)
```

### 🔧 Techniques Used
- ✅ Mixed Precision Training
- ✅ Paired supervised learning
- ✅ Checkpoint saving every 5–10 epochs

### 📊 Results & Evaluation
| Metric | Description |
|--------|-------------|
| **SSIM** | Structural Similarity Index |
| **PSNR** | Peak Signal-to-Noise Ratio |
| **Visual** | Input vs Generated vs Ground Truth |

---

## 🔄 Question 3: Unpaired Image Translation using CycleGAN

### 🎯 Objective
Unpaired image-to-image translation using CycleGAN:
- **Sketch → Photo**
- **Photo → Sketch**
- Structural consistency via cycle constraints

### 📦 Datasets
| Dataset | Link |
|---------|------|
| ✏️ TU-Berlin Sketch | [HuggingFace](https://huggingface.co/datasets/sdiaeyu6n/tu-berlin) |
| 🖼️ Sketchy Dataset | [Kaggle](https://www.kaggle.com/datasets/sharanyasundar/sketchy-dataset) |
| ✍️ Google QuickDraw | [Kaggle](https://www.kaggle.com/c/quickdraw-doodle-recognition/data) |

### 🏗️ Architecture

#### Generators (ResNet-based)
| Generator | Task |
|-----------|------|
| `G_AB` | Sketch → Photo |
| `G_BA` | Photo → Sketch |

- 6 ResNet Blocks (optimized for Kaggle)
- Image Size: 128 × 128

#### Discriminators (PatchGAN)
| Discriminator | Domain |
|---------------|--------|
| `D_A` | Classifies Sketch domain |
| `D_B` | Classifies Photo domain |

### ⚙️ Training Config
```python
IMG_SIZE   = 128
BATCH_SIZE = 4–8
LR         = 0.0002
BETAS      = (0.5, 0.999)
N_RESBLOCKS = 6
```

### 📉 Loss Functions
```
Total Loss = Adversarial Loss + Cycle Consistency Loss + Identity Loss
```

### 🔧 Techniques Used
- ✅ Mixed Precision Training
- ✅ Dual GPU (T4 × 2)
- ✅ Frequent checkpointing
- ✅ Dataset subset for memory efficiency

### 📊 Results & Evaluation
| Metric | Description |
|--------|-------------|
| **SSIM** | Structural Similarity Index |
| **PSNR** | Peak Signal-to-Noise Ratio |
| **Visual** | Input → Translated → Reconstructed |

---

## 🚀 App Deployment

### CycleGAN Gradio App (`app.py`)
A real-time **Sketch → Photo** conversion app built with Gradio.

#### Features
- Upload any sketch image
- Instantly generates a realistic photo using the trained CycleGAN (`G_BA`)
- Clean, simple UI

#### Run Locally
```bash
pip install gradio torch torchvision pillow
python app.py
```

> **Note:** Place the trained checkpoint `cyc_G_BA_ep50.pth` in the same directory before running.

---

## 🛠️ Environment Setup

| Tool | Version / Details |
|------|-------------------|
| Platform | [Kaggle](https://www.kaggle.com/) |
| GPU | T4 × 2 (Dual GPU) |
| Framework | PyTorch |
| Mixed Precision | `torch.cuda.amp` |
| App Framework | Gradio |

### Install Dependencies
```bash
pip install torch torchvision gradio scikit-image pillow numpy matplotlib
```

---

## 📌 Notes
- All models trained on Kaggle with GPU T4 × 2
- Dataset subsets used where needed to fit GPU memory
- Checkpoints saved every 5–10 epochs
- Mixed precision used throughout for speed and memory efficiency

---

*AI4009 — Generative AI | Spring 2026 | FAST-NUCES*
