
import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, length = x.size()
        out = F.avg_pool1d(x, length).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1)
        return x * out

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonlinearity, se, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonlinear = nonlinearity
        self.se = se
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv1d(in_channels, exp_size, 1, 1, 0, bias=False),
            nn.BatchNorm1d(exp_size),
            nonlinearity,

            # Depthwise
            nn.Conv1d(exp_size, exp_size, kernal_size, stride, padding, groups=exp_size, bias=False),
            nn.BatchNorm1d(exp_size),
            self.se(exp_size) if se else nn.Identity(),
            nonlinearity,

            # Pointwise linear
            nn.Conv1d(exp_size, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3_1D(nn.Module):
    def __init__(self, n_in_channels=1, output_dim=20):
        super(MobileNetV3_1D, self).__init__()
        
        # Adapting typical MobileNetV3 Small structure for 1D and smaller input size
        # Input size is [Batch, 1, 20]
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_in_channels, 16, 3, 2, 1, bias=False), # Stride 2 -> Length 10
            nn.BatchNorm1d(16),
            h_swish()
        )

        self.layers = nn.ModuleList([
            # in, out, k, s, non_linear, se, exp
            MobileBlock(16, 16, 3, 2, nn.ReLU(inplace=True), True, 16), # Length 5
            MobileBlock(16, 24, 3, 2, nn.ReLU(inplace=True), False, 72), # Length 3
            MobileBlock(24, 24, 3, 1, nn.ReLU(inplace=True), False, 88), # Length 3
            MobileBlock(24, 40, 5, 2, h_swish(), True, 96), # Length 2
            MobileBlock(40, 40, 5, 1, h_swish(), True, 240), # Length 2
            MobileBlock(40, 40, 5, 1, h_swish(), True, 240), # Length 2
            MobileBlock(40, 48, 5, 1, h_swish(), True, 120), # Length 2
            MobileBlock(48, 48, 5, 1, h_swish(), True, 144), # Length 2
            MobileBlock(48, 96, 5, 2, h_swish(), True, 288), # Length 1 (Approx)
        ])
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm1d(576),
            h_swish()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
