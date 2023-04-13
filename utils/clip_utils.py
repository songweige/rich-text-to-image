import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import clip

### from: https://github.com/GaParmar/clean-fid/blob/c8ffa420a3923e8fd87c1e75170de2cf59d2644b/cleanfid/clip_features.py
def img_preprocess_clip(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
    ])
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)


### from: https://github.com/GaParmar/clean-fid/blob/c8ffa420a3923e8fd87c1e75170de2cf59d2644b/cleanfid/clip_features.py
class CLIP_fx():
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
        self.name = "clip_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, img_t):
        img_x = img_t/255.0
        T_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        B,C,H,W = img_x.shape
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z

    def encode_text(self, text_t):
        text = clip.tokenize(text_t).to('cuda')
        z_text = self.model.encode_text(text).float()
        return z_text


class CLIPEncoder(nn.Module):
    def __init__(self, clip_version='ViT-B/32', pretrained='', device='cuda'):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == 'ViT-H-14':
                self.pretrained = 'laion2b_s32b_b79k'
            elif self.clip_version == 'ViT-g-14':
                self.pretrained = 'laion2b_s12b_b42k'
            else:
                self.pretrained = 'openai'

        self.model = CLIP_fx(name=clip_version)

        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if not isinstance(text, (list, tuple)):
            text = [text]

        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        if isinstance(image, str):  # filenmae
            # import ipdb;ipdb.set_trace()
            image = torchvision.io.read_image(image) # THWC
        image = img_preprocess_clip(image)
        image = torch.from_numpy(image).cuda()
        image = image.permute(2, 0, 1).type(torch.float32)
        image_features = self.model(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity