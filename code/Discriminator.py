import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(Discriminator, self).__init__()
        # [Conv -> LeakyReLU] * N с чередованием stride=1 и stride=2 для уменьшения размера
        self.features = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, 2, 1), nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 1, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 2, 3, 2, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 4, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x