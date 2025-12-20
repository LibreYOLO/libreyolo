"""
Neural network architecture for Libre YOLO8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Converts the distribution of 4*reg_max channels into the final 4 coordinates (x, y, w, h).
    """
    def __init__(self, c1=16):
        super().__init__()
        # 1x1 convolution with fixed weights 0..c1-1 to calculate the expectation
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # Input x: (Batch, 64, H, W) for reg_max=16
        b, c, h, w = x.shape
        # Reshape to (Batch, 4, 16, H, W) and transpose to (Batch, 16, 4, H, W)
        x = x.view(b, 4, self.c1, h, w).transpose(2, 1)
        # Apply softmax to get probability distribution
        x = F.softmax(x, dim=1)
        # Apply the convolution (weighted sum) to reduce 16 channels to 1
        # Reshape to (Batch, 16, 4*H, W) so Conv2d sees a valid 4D tensor
        x = self.conv(x.reshape(b, self.c1, 4 * h, w))
        # Reshape back to (Batch, 4, H, W)
        return x.view(b, 4, h, w)


class Conv(nn.Module):
    """Standard convolution block: Conv2d + BatchNorm + SiLU"""
    def __init__(self, c_out, c_in, k, s, p):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=c_out)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.batchnorm(x)
        x = self.silu(x)
        return x


class Bottleneck(nn.Module):
    """Residual bottleneck block"""
    def __init__(self, c_out, c_in, res_connection):
        super().__init__()
        self.conv1 = Conv(c_out=c_out, c_in=c_in, k=3, s=1, p=1)
        self.conv2 = Conv(c_out=c_out, c_in=c_out, k=3, s=1, p=1)
        self.res_connection = res_connection
        
    def forward(self, x):
        if self.res_connection:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class C2F(nn.Module):
    """C2f module with split-concat-bottleneck structure"""
    def __init__(self, c_out, c_in, res_connection, nb_bottlenecks):
        super().__init__()
        self.conv1 = Conv(c_out=c_out, c_in=c_in, k=1, s=1, p=0)
        # Calculate input channels for the final convolution based on concatenation size
        self.conv2 = Conv(c_out=c_out, c_in=int((nb_bottlenecks + 2) * c_out / 2), k=1, s=1, p=0)
        
        self.bottlenecks = nn.ModuleList([
            Bottleneck(c_out=c_out//2, c_in=c_out//2, res_connection=res_connection) 
            for _ in range(nb_bottlenecks)
        ])
        
    def forward(self, x):
        x = self.conv1(x)
        batch, c, h, w = x.shape
        
        # Split features
        x2 = x[:, c//2:, :, :]
        
        # Pass through bottlenecks and concatenate results
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)
            x = torch.cat((x, x2), dim=1)
        
        x = self.conv2(x)
        return x


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast"""
    def __init__(self, c_out, c_in):
        super().__init__()
        self.conv1 = Conv(c_out=c_out//2, c_in=c_in, k=1, s=1, p=0)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = Conv(c_out=c_out, c_in=4*(c_out//2), k=1, s=1, p=0)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.conv2(x)
        return x


class Backbone(nn.Module):
    """Feature extraction backbone"""
    def __init__(self, config):
        super().__init__()
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
                
        self.p1 = Conv(c_out=int(64*w), c_in=3, k=3, s=2, p=1)
        self.p2 = Conv(c_out=int(128*w), c_in=int(64*w), k=3, s=2, p=1)
        self.c2f1 = C2F(c_out=int(128*w), c_in=int(128*w), res_connection=True, nb_bottlenecks=max(1, int(round(3*d))))
        
        self.p3 = Conv(c_out=int(256*w), c_in=int(128*w), k=3, s=2, p=1)
        self.c2f2 = C2F(c_out=int(256*w), c_in=int(256*w), res_connection=True, nb_bottlenecks=max(1, int(round(6*d))))
        
        self.p4 = Conv(c_out=int(512*w), c_in=int(256*w), k=3, s=2, p=1)
        self.c2f3 = C2F(c_out=int(512*w), c_in=int(512*w), res_connection=True, nb_bottlenecks=max(1, int(round(6*d))))
        
        self.p5 = Conv(c_out=int(512*w*r), c_in=int(512*w), k=3, s=2, p=1)
        self.c2f4 = C2F(c_out=int(512*w*r), c_in=int(512*w*r), res_connection=True, nb_bottlenecks=max(1, int(round(3*d))))
        
        self.sppf = SPPF(c_out=int(512*w*r), c_in=int(512*w*r))
        
    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.c2f1(x)
        x = self.p3(x)
        x = self.c2f2(x)
        x8 = x.clone()  # P3 (Stride 8)
        
        x = self.p4(x)
        x = self.c2f3(x)
        x16 = x.clone()  # P4 (Stride 16)
        
        x = self.p5(x)
        x = self.c2f4(x)
        x32 = self.sppf(x)  # P5 (Stride 32)
        
        return x8, x16, x32


class Neck(nn.Module):
    """Feature pyramid network neck"""
    def __init__(self, config):
        super().__init__()
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')        
        self.c2f21 = C2F(c_out=int(512*w), c_in=int(512*w*(1 + r)), res_connection=False, nb_bottlenecks=max(1, int(round(3*d))))
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f11 = C2F(c_out=int(256*w), c_in=int(768*w), res_connection=False, nb_bottlenecks=max(1, int(round(3*d))))    
        
        self.conv1 = Conv(c_out=int(256*w), c_in=int(256*w), k=3, s=2, p=1)
        self.c2f12 = C2F(c_out=int(512*w), c_in=int(768*w), res_connection=False, nb_bottlenecks=max(1, int(round(3*d))))
        
        self.conv2 = Conv(c_out=int(512*w), c_in=int(512*w), k=3, s=2, p=1)
        self.c2f22 = C2F(c_out=int(512*w*r), c_in=int(512*w*(1 + r)), res_connection=False, nb_bottlenecks=max(1, int(round(3*d))))
        
    def forward(self, x8, x16, x32):
        # 1. Upsample P5 (x32) to P4 size
        aux1 = self.upsample1(x32)
        x16 = torch.cat((aux1, x16), dim=1) 
        x16 = self.c2f21(x16)
        
        # 2. Upsample P4 (x16) to P3 size
        aux2 = self.upsample2(x16)
        x8 = torch.cat((aux2, x8), dim=1)   
        x8 = self.c2f11(x8)
        
        # 3. Downsample P3 (x8) to P4 size
        aux3 = self.conv1(x8)
        x16 = torch.cat((aux3, x16), dim=1) 
        x16 = self.c2f12(x16)
        
        # 4. Downsample P4 (x16) to P5 size
        aux4 = self.conv2(x16)
        x32 = torch.cat((aux4, x32), dim=1) 
        x32 = self.c2f22(x32)
        
        return x8, x16, x32


class Head(nn.Module):
    """Decoupled detection head"""
    def __init__(self, c_in, c_box, c_cls, reg_max, nb_classes):
        super().__init__()
        
        # Decoupled Head: Two branches with different widths
        # Box Branch: c_in -> c_box -> c_box -> 4*reg_max
        self.conv11 = Conv(c_out=c_box, c_in=c_in, k=3, s=1, p=1)
        self.conv12 = Conv(c_out=c_box, c_in=c_box, k=3, s=1, p=1)
        
        # Class Branch: c_in -> c_cls -> c_cls -> nb_classes
        self.conv21 = Conv(c_out=c_cls, c_in=c_in, k=3, s=1, p=1)
        self.conv22 = Conv(c_out=c_cls, c_in=c_cls, k=3, s=1, p=1)
        
        self.cnn1 = nn.Conv2d(in_channels=c_box, out_channels=4*reg_max, kernel_size=1, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=c_cls, out_channels=nb_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        box = self.cnn1(self.conv12(self.conv11(x)))
        cls = self.cnn2(self.conv22(self.conv21(x)))
        return box, cls


class LibreYOLO8Model(nn.Module):
    """Main Libre YOLO model"""
    def __init__(self, config, reg_max, nb_classes):
        super().__init__()
        
        self.configuration = {
            'n': {'d': 0.33, 'w': 0.25, 'r': 2.0},
            's': {'d': 0.33, 'w': 0.50, 'r': 2.0},
            'm': {'d': 0.67, 'w': 0.75, 'r': 1.5},
            'l': {'d': 1.00, 'w': 1.00, 'r': 1.0},
            'x': {'d': 1.00, 'w': 1.25, 'r': 1.0}
        }

        d = self.configuration[config]['d']
        w = self.configuration[config]['w']
        r = self.configuration[config]['r']
        
        self.backbone = Backbone(config=config)
        self.neck = Neck(config=config)
        
        # --- HEAD CONFIGURATION ---
        # Channels P3 = 256 * w
        c_p3 = int(256 * w)
        
        # Box channels: max(64, c_p3 // 4)
        c_box = max(64, c_p3 // 4)
        
        # Cls channels: max(c_p3, min(nb_classes, 100))
        c_cls = max(c_p3, min(nb_classes, 100))
        
        self.head8 = Head(c_in=int(256*w), c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head16 = Head(c_in=int(512*w), c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        self.head32 = Head(c_in=int(512*w*r), c_box=c_box, c_cls=c_cls, reg_max=reg_max, nb_classes=nb_classes)
        
        self.dfl = DFL(c1=reg_max)

    def forward(self, x):
        x8, x16, x32 = self.backbone(x)
        x8, x16, x32 = self.neck(x8, x16, x32)
        
        box8, cls8 = self.head8(x8)
        box16, cls16 = self.head16(x16)
        box32, cls32 = self.head32(x32)
        
        decoded_box8 = self.dfl(box8)
        decoded_box16 = self.dfl(box16)
        decoded_box32 = self.dfl(box32)
        
        out = {
            'x8': {'box': decoded_box8, 'cls': cls8},
            'x16': {'box': decoded_box16, 'cls': cls16},
            'x32': {'box': decoded_box32, 'cls': cls32}
        }
        
        return out

