import torch
from torch import nn

class DoubleConvolution(nn.Module):
    """
        2 Convolution layers back to back with ReLU layer as a sandwich
    """

    def __init__(self, in_channels : int, out_channels : int) -> None:
        super().__init__() # type: ignore
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Downsample(nn.Module):
    """
        Downsampling the vector using Double Convolutions and Max Pooling
    """
    
    def __init__(self, in_channels : int, out_channels : int) -> None:
        super().__init__()  # type: ignore

        self.conv = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        down = self.conv(x)
        return down, self.pool(down)

class UpSample(nn.Module):
    """
        Upsampling the vector using Transpose Convolution and skip connections
    """

    def __init__(self, in_channels : int, out_channels : int) -> None:
        super().__init__()  # type: ignore

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2,
                               kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x : torch.Tensor, skip : torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.concat([x, skip])
        return self.conv(x)

class UNet(nn.Module):
    """
        UNet Architecture        
    """

    def __init__(self, in_channels : int, n_classes : int) -> None:
        super().__init__()  # type: ignore
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.down_sample1 = Downsample(in_channels=in_channels, out_channels=64)
        self.down_sample2 = Downsample(in_channels=64, out_channels=128)
        self.down_sample3 = Downsample(in_channels=128, out_channels=256)
        self.down_sample4 = Downsample(in_channels=256, out_channels=512)

        self.feature_map = DoubleConvolution(in_channels=512, out_channels=1024)

        self.up_sample1 = UpSample(in_channels=1024, out_channels=512)
        self.up_sample2 = UpSample(in_channels=512, out_channels=256)
        self.up_sample3 = UpSample(in_channels=256, out_channels=128)
        self.up_sample4 = UpSample(in_channels=128, out_channels=64)

        self.collapse = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        down1, p1 = self.down_sample1(x)
        down2, p2 = self.down_sample1(p1)
        down3, p3 = self.down_sample1(p2)
        down4, p4 = self.down_sample1(p3)

        feature_vec = self.feature_map(p4)

        up1 = self.up_sample1(feature_vec, down4)
        up2 = self.up_sample1(up1, down3)
        up3 = self.up_sample1(up2, down2)
        up4 = self.up_sample1(up3, down1)

        return self.collapse(up4)
