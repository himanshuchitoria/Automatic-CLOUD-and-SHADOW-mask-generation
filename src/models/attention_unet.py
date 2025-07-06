# src/models/attention_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BatchNorm -> ReLU (x2)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Attention Gate for U-Net skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpBlock(nn.Module):
    """Upsampling block with attention and skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed (for odd input sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=features[3], F_l=features[3], F_int=features[2])
        self.decoder4 = ConvBlock(features[3]*2, features[2])

        self.up3 = nn.ConvTranspose2d(features[2], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=features[2], F_l=features[2], F_int=features[1])
        self.decoder3 = ConvBlock(features[2]*2, features[1])

        self.up2 = nn.ConvTranspose2d(features[1], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=features[1], F_l=features[1], F_int=features[0])
        self.decoder2 = ConvBlock(features[1]*2, features[0])

        self.up1 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features[0], features[0])

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Decoder with attention
        up4 = self.up4(b)
        att4 = self.att4(g=up4, x=e4)
        d4 = self.decoder4(torch.cat([att4, up4], dim=1))

        up3 = self.up3(d4)
        att3 = self.att3(g=up3, x=e3)
        d3 = self.decoder3(torch.cat([att3, up3], dim=1))

        up2 = self.up2(d3)
        att2 = self.att2(g=up2, x=e2)
        d2 = self.decoder2(torch.cat([att2, up2], dim=1))

        up1 = self.up1(d2)
        d1 = self.decoder1(up1)

        out = self.final_conv(d1)
        return out

# Example usage
if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, out_channels=3)
    x = torch.randn(2, 3, 256, 256)  # Batch of 2, 3-band, 256x256 images
    y = model(x)
    print("Output shape:", y.shape)  # Should be (2, 3, 256, 256)
