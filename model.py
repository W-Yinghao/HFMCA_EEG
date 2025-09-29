import pdb
import random

from itertools import chain, combinations
from scipy.signal import firwin

import torch
import torch.nn as nn
import torch.nn.functional as F

# The implementation of the encoder architecture for the SEED dataset is referenced from https://github.com/bohu615/corticomuscular-eigen-encoder
# The implementation of the encoder architecture for the BCI-IV 2A dataset is reimplemented based on the description of  <<Self-supervised contrastive learning for EEG-based cross-subject motor imagery recognition>> 


class LAST_CNN(nn.Module):
    def __init__(self, in_channels = 256, HIDDEN = 200, out_channels = 64, sample_time = 4):
        super(LAST_CNN, self).__init__()
        
        self.cnn_list = []
        self.bn_list = []

        self.dim = out_channels

        self.cnn_list.append(nn.Conv1d(in_channels, HIDDEN, kernel_size = 1, stride = 1, padding = 0,  bias=True))
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.cnn_list.append(nn.Conv1d(HIDDEN, HIDDEN, kernel_size = sample_time, stride = 1, padding = 0,  bias=True))
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.cnn_list = nn.ModuleList(self.cnn_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.cnn_final = nn.Conv1d(HIDDEN, out_channels, kernel_size = 1, stride = 1, padding = 0, bias=True)

    def forward(self, x):

        for i in range(0, len(self.cnn_list)):
            x = self.cnn_list[i](x)
            x = torch.relu(x)
            x = self.bn_list[i](x)
            
        x = self.cnn_final(x)
        x = torch.sigmoid(x)

        return x


class NETWORK_F_MLP(nn.Module):
    def __init__(self, input_dim = 784, HIDDEN = 200, out_dim = 200, how_many_layers = 2):
        super(NETWORK_F_MLP, self).__init__()
        self.dim = out_dim
        self.many_layer = how_many_layers

        self.fc_list = []
        self.bn_list = []

        self.fc_list.append(nn.Linear(input_dim, HIDDEN, bias=True))
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        for i in range(0, self.many_layer-1):
            self.fc_list.append(nn.Linear(HIDDEN, HIDDEN, bias=True))
            self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.fc_list = nn.ModuleList(self.fc_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.fc_final = nn.Linear(HIDDEN, out_dim, bias=True)

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)

        for i in range(0, self.many_layer):
            x = self.fc_list[i](x)
            x = torch.relu(x)
            x = self.bn_list[i](x)

        x = self.fc_final(x)
        x = torch.sigmoid(x)
        return x


class Advanced1DCNN_channel(nn.Module):
    def __init__(self, dim, fc1_dim=256, f_mlp_dim=7936, input_channel=1, num_classes=64, input_size=4000, num_channel=60):
        super(Advanced1DCNN_channel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(fc1_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.fc3 = nn.Linear(512, 128)

        self.MLP = NETWORK_F_MLP(input_dim = f_mlp_dim, HIDDEN = 4000, out_dim = 256, how_many_layers = 1)

    def forward(self, x):
        bs, channel = x.shape[0], x.shape[1]
        x = x.unsqueeze(2)
        x = x.flatten(0, 1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        out = out.reshape(bs, channel, -1)
        out = out.flatten(-2, -1)
        out = self.MLP(out)
        return out

class Baseline_seed(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        
        self.n_dim = n_dim
        self.encoder = Advanced1DCNN_channel(n_dim)

        self.fc = nn.Sequential(
            nn.Linear(256, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sample_time = 4
        self.final_net = LAST_CNN(self.n_dim, self.n_dim * 4, self.n_dim, self.sample_time)
        # self.final_net = LAST_MLP(self.n_dim * self.sample_time, 1024, self.n_dim, self.sample_time)

    def forward(self, x, train=False):

        if train:
            sample_t = x.shape[1]
            if self.sample_time != sample_t:
                x = x[:,:self.sample_time, :, :]
            x = x.flatten(0, 1)

        bs = x.shape[0] // self.sample_time
        
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        multi_feature = x.squeeze(-1).squeeze(-1)
        multi_feature = torch.stack([multi_feature[bs*k : bs*(k+1)] for k in range(0, self.sample_time)])
        multi_feature = multi_feature.permute(1, 2, 0)

        x_h = self.final_net(multi_feature).squeeze(-1)

        if train:
            return multi_feature, x_h
        else:
            return x, x_h


"""

Temporal filtering: Conv → SENet → BN

Spatial filtering: Conv → SENet → BN → ELU → AvgPool(1×8) → Dropout(0.2)

Feature compression: Separable Conv → SENet → BN → ELU → AvgPool(1×16) → Dropout(0.2)

Classifier: Flatten → Dense(32) → ELU → Dropout(0.2) → Dense(c) → Softmax


"""

# SENet block (Squeeze-and-Excitation)
class SENet(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SENet, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)  # pool over T
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, T, C)
        
        # Permute to (N, C, T) for pooling
        x_perm = x # .permute(0, 2, 1)   # (N, C, T)
        
        # Global average pooling over T -> (N, C, 1)
        w = self.global_avgpool(x_perm) # .squeeze(-1)  
        w = w.squeeze(-1)  # (N, C)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        # print(w.shape)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w).unsqueeze(-1)  # (N, 1, C)
        
        # Scale: broadcast along T
        # print(x.shape, w.shape)
        return x * w  # (N, T, C)



# Temporal Filtering Module
class TemporalFiltering(nn.Module):
    def __init__(self, in_channels, kernel_size=(1,128), out_channels=8):
        super(TemporalFiltering, self).__init__()
        # Conv2d expects (N, Cin, H, W)
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels,
            kernel_size=kernel_size, padding=(0, kernel_size[1]//2), bias=False
        )
        self.se = SENet(out_channels, reduction=2)  # SENet on 8 feature maps
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, C, T)
        x = x.unsqueeze(1)  # -> (N, 1, C, T)

        # Temporal convolution: (N, 8, C, T)
        x = self.conv(x)

        # x_perm = x.permute(0, 2, 3, 1)  # (N, C, T, 8)

        # Reshape for SENet: (N, T, C') where C' = out_channels=8
        # x_perm = x.permute(0, 3, 1, 2)   # (N, T, 8, C)
        N, F , C, T= x.shape
        x_se = x.reshape(N, F, C*T) # merge spatial C into features
        x_se = self.se(x_se)             # apply SENet over feature maps
        x_se = x_se.view(N, F, C, T) # .permute(0, 3, 1, 2) # back to (N, 8, C, T)

        # BatchNorm over feature map dimension (channel=8)
        x_out = self.bn(x_se)

        return x_out  # (N, 8, C, T)


# Spatial Filtering Module
class SpatialFiltering(nn.Module):
    """
    Spatial filtering module for MI signals.
    
    Input: x of shape (N, 8, C, T)
    Process:
        - Depthwise convolution with kernel (C,1), D=2 → 16 channels
        - SENet attention over channels
        - BatchNorm over channels
        - ELU activation
        - AvgPool (1,8) over temporal dimension
        - Dropout for regularization
    Output: (N, 16, 1, T//8)
    """
    def __init__(self, in_channels=8, depth_multiplier=2, num_spatial=22):
        super(SpatialFiltering, self).__init__()
        out_channels = in_channels * depth_multiplier  # 16
        
        # Depthwise convolution: groups=in_channels ensures 2 filters per input channel
        self.depthwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(num_spatial,1), groups=in_channels, bias=False
        )
        self.se = SENet(out_channels, reduction=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1,8))
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        # x: (N, 8, C, T)
        x = self.depthwise(x)      # -> (N,16,1,T)
        N, F, C, T = x.shape
        x_se = x.reshape(N, F, C * T)
        x_se = self.se(x_se)
        x = x_se.view(N, F, C, T)  # SENet on channels
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)           # -> (N,16,1,T//8)
        x = self.drop(x)
        return x



# Feature Compression Module
class FeatureCompression(nn.Module):
    """
    Feature compression module for MI signals.
    
    Input: x of shape (N, 16, 1, T//8)
    Process:
        - Depthwise separable convolution (16 depthwise + 16 pointwise)
        - SENet attention over channels
        - BatchNorm over channels
        - ELU activation
        - AvgPool (1,16) over temporal dimension
        - Dropout for regularization
    Output: (N, 16, 1, T//128)
    """
    def __init__(self, in_channels=16, depthwise_kernel=(1,32), pointwise_out=16):
        super(FeatureCompression, self).__init__()
        
        # Depthwise convolution: groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=depthwise_kernel, groups=in_channels, bias=False
        )
        # Pointwise convolution: combines channels
        self.pointwise = nn.Conv2d(in_channels, pointwise_out, kernel_size=(1,1), bias=False)
        
        self.se = SENet(pointwise_out, reduction=2)
        self.bn = nn.BatchNorm2d(pointwise_out)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1,16))
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        # print(x.shape)
        # x: (N,16,1,T//8)
        x = self.depthwise(x)      # -> (N,16,1,T//8)
        x = self.pointwise(x)      # -> (N,16,1,T//8)
        # print(x.shape)

        N, F, C, T = x.shape
        x_se = x.reshape(N, F, C * T)
        x_se = self.se(x_se)
        # print(x_se.shape)
        x = x_se.view(N, F, C, T)  # SENet on channels
        # print(x.shape)
        
        x = self.bn(x)
        x = self.elu(x)
        # print(x.shape)
        x = self.pool(x)           # -> (N,16,1,T//128)
        x = self.drop(x)
        return x


# Final Classifier
class EEGNetLike(nn.Module):
    def __init__(self, num_channels, num_classes, time_steps):
        super(EEGNetLike, self).__init__()
        self.temporal = TemporalFiltering(in_channels=num_channels)
        self.spatial = SpatialFiltering(in_channels=8, num_spatial=num_channels)
        self.compress = FeatureCompression(in_channels=16)  # depends on spatial output

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.compress(x)

        return x.squeeze(-1).squeeze(-1)


class Baseline_bci4_v2(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.n_dim = n_dim

        # self.encoder = Advanced1DCNN_channel(n_dim, fc1_dim=768, f_mlp_dim=2816)
        self.encoder = EEGNetLike(num_channels=22, num_classes=4, time_steps=1001)

        self.fc = nn.Sequential(
            nn.Linear(80, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

        self.sample_time = 4
        self.final_net = LAST_CNN(self.n_dim, self.n_dim * 4, self.n_dim, self.sample_time)
        # self.final_net = LAST_MLP(self.n_dim * self.sample_time, 1024, self.n_dim, self.sample_time)

    def forward(self, x, train=False):

        if train:
            sample_t = x.shape[1]
            if self.sample_time != sample_t:
                x = x[:,:self.sample_time, :, :]
            x = x.flatten(0, 1)
        bs = x.shape[0] // self.sample_time
        
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        multi_feature = x.squeeze(-1).squeeze(-1)
        multi_feature = torch.stack([multi_feature[bs*k:bs*(k+1)] for k in range(0, self.sample_time)])
        multi_feature = multi_feature.permute(1, 2, 0)

        x_h = self.final_net(multi_feature).squeeze(-1)

        if train:
            return multi_feature, x_h
        else:
            return x, x_h
        
