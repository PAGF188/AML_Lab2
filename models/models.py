import torchvision.models as models
from torch import nn
import torch
from config import N_CLASES, N_CLASES_UNET
import torch.nn.functional as F
import pdb

def denseNet121_pretrained():
    # Pre-trained on ImageNET
    model = models.densenet121(pretrained=True)
    

    # DenseNet121 has only one linear layer in the clasiffier
    # Change final linear layer to 10 outpus
    model.classifier = nn.Linear(1024, N_CLASES)
    #print(model)
    return model


def denseNet121_basic():
    # NOT pre-trained
    model = models.densenet121(pretrained=False)
    
    # DenseNet121 has only one linear layer in the clasiffier
    # Change final linear layer to 10 outpus
    model.classifier = nn.Linear(1024, N_CLASES)
    #print(model)
    return model


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=N_CLASES_UNET, init_features=64):
        super(UNet, self).__init__()
        features = init_features

        # DOWN
        self.encoder1 = self.DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = self.DoubleConv(features, features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.encoder3 = self.DoubleConv(features*2, features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.encoder4 = self.DoubleConv(features*4, features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # HORIZONTAL
        self.horizontal = self.DoubleConv(features*8, features*16)

        # UP
        self.upconv1 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder1 = self.DoubleConv((features*8)*2, features*8)
        self.upconv2 = nn.ConvTranspose2d(features*8 , features*4, kernel_size=2, stride=2)
        self.decoder2 = self.DoubleConv((features*4)*2, features*4)
        self.upconv3 = nn.ConvTranspose2d(features*4 , features*2, kernel_size=2, stride=2)
        self.decoder3 = self.DoubleConv((features*2)*2, features*2)
        self.upconv4 = nn.ConvTranspose2d(features*2 , features, kernel_size=2, stride=2)
        self.decoder4 = self.DoubleConv((features)*2, features)

        # 1x1 final convolution
        self.outConv = nn.Conv2d(features, out_channels, kernel_size=1)



    def forward(self, x):
        # DOWN
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # HORIZONTAL
        horizontal = self.horizontal(self.pool4(enc4))

        # UP
        dec1 = self.upconv1(horizontal)
        diffY = enc4.size()[2] - dec1.size()[2]; diffX = enc4.size()[3] - dec1.size()[3]
        # Instead of cropping -> padding
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec1 = torch.cat([enc4, dec1], dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.upconv2(dec1)
        diffY = enc3.size()[2] - dec2.size()[2]; diffX = enc3.size()[3] - dec2.size()[3]
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec2 = torch.cat([enc3, dec2], dim=1)
        dec2 = self.decoder2(dec2)

        dec3 = self.upconv3(dec2)
        diffY = enc2.size()[2] - dec3.size()[2]; diffX = enc2.size()[3] - dec3.size()[3]
        dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec3 = torch.cat([enc2, dec3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec4 = self.upconv4(dec3)
        diffY = enc1.size()[2] - dec4.size()[2]; diffX = enc1.size()[3] - dec4.size()[3]
        dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec4 = torch.cat([enc1, dec4], dim=1)
        dec4 = self.decoder4(dec4)
        
        final = self.outConv(dec4)
        return final

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, features):
            super().__init__()
            # Instead of pad==, 1 -> conseguir mism size en salida.
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)
