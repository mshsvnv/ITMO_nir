import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg11(weights=VGG11_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:20]).eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
        # Нормализация ImageNet
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)