import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import OrderedDict

# Define model architecture
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c,c,3,bias=False), 
            nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c,c,3,bias=False), 
            nn.InstanceNorm2d(c))
    def forward(self, x): return x + self.block(x)

class ResNetGen(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_res=6):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(in_c,ngf,7,bias=False),
             nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for m in [1,2]:
            L += [nn.Conv2d(ngf*m,ngf*m*2,3,stride=2,padding=1,bias=False),
                  nn.InstanceNorm2d(ngf*m*2), nn.ReLU(True)]
        for _ in range(n_res): L.append(ResBlock(ngf*4))
        for m in [2,1]:
            L += [nn.ConvTranspose2d(ngf*m*2,ngf*m,3,stride=2,padding=1,
                                     output_padding=1,bias=False),
                  nn.InstanceNorm2d(ngf*m), nn.ReLU(True)]
        L += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,out_c,7), nn.Tanh()]
        self.model = nn.Sequential(*L)
    def forward(self, x): return self.model(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = ResNetGen().to(device)

# FIX: Remove 'module.' prefix from DataParallel saved model
checkpoint = torch.load('cyc_G_BA_ep50.pth', map_location=device)

# Remove "module." from keys if present
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    else:
        new_state_dict[k] = v

# Load with strict=False to be safe
G.load_state_dict(new_state_dict, strict=True)
G.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

def sketch_to_photo(sketch):
    img = transform(sketch).unsqueeze(0).to(device)
    with torch.no_grad():
        photo = G(img)
    return denorm(photo[0].cpu()).permute(1,2,0).numpy()

# Gradio interface
iface = gr.Interface(
    fn=sketch_to_photo,
    inputs=gr.Image(type="pil", label="Upload Sketch"),
    outputs=gr.Image(label="Generated Photo"),
    title="Sketch to Photo - CycleGAN",
    description="Convert your sketch into a realistic photo!"
)

iface.launch()