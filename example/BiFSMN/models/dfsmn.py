import paddle.nn as nn
import paddle.nn.functional as F
from .utils import weight_init


class DfsmnLayer(nn.Layer):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(
            nn.Linear(backbone_memory_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, backbone_memory_size),
            nn.Dropout(dropout)
        )
        self.memory = nn.Conv1D(in_channels=backbone_memory_size,
                                out_channels=backbone_memory_size,
                                kernel_size=left_kernel_size + right_kernel_size + 1,
                                padding=0,
                                stride=1,
                                dilation=dilation,
                                groups=backbone_memory_size)

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B, N, T + (l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out.transpose([0, 2, 1]))  # (B, T, N)
        output = fc_output.transpose([0, 2, 1]) + residual  # (B, N, T)
        return output


class DfsmnLayerBN(nn.Layer):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 dropout=0.0):
        super().__init__()
        self.fc_trans = nn.Sequential(
            nn.Conv1D(backbone_memory_size, hidden_size, 1),
            nn.BatchNorm1D(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1D(hidden_size, backbone_memory_size, 1),
            nn.BatchNorm1D(backbone_memory_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.memory = nn.Sequential(
            nn.Conv1D(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1D(backbone_memory_size),
            nn.ReLU(),
        )

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B, N, T + (l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        # fc-transform
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        output = fc_output + residual  # (B, N, T)
        return output


class DfsmnModel(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=6,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.2,
                 dfsmn_with_bn=True,
                 distill=False,
                 **kwargs):
        super().__init__()
        self.front_end = nn.Sequential(
            nn.Conv2D(in_channels,
                      out_channels=frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2, frondend_kernel_size // 2)),
            nn.BatchNorm2D(frondend_channels),
            nn.ReLU(),
            nn.Conv2D(frondend_channels,
                      out_channels=2 * frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2, frondend_kernel_size // 2)),
            nn.BatchNorm2D(2 * frondend_channels),
            nn.ReLU()
        )
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=2 * frondend_channels * self.n_mels // 4,
                      out_features=backbone_memory_size),
            nn.ReLU(),
        )
        backbone = list()
        for idx in range(num_layer):
            if dfsmn_with_bn:
                backbone.append(
                    DfsmnLayerBN(hidden_size, backbone_memory_size,
                                 left_kernel_size, right_kernel_size, dilation,
                                 dropout))
            else:
                backbone.append(
                    DfsmnLayer(hidden_size, backbone_memory_size,
                               left_kernel_size, right_kernel_size, dilation,
                               dropout))
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_memory_size * self.n_mels // 4, num_classes),
        )
        self.distill = distill
        self.apply(weight_init)

    def forward(self, input_feat):
        # input_feat: B, 1, N, T
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.reshape([batch, -1, out.shape[3]]).transpose([0, 2, 1])  # B, T, N1
        out = self.fc1(out).transpose([0, 2, 1])  # B, N, T
        features = []
        for layer in self.backbone:
            out = layer(out)
            features.append(out)
        
        out = out.reshape([batch, -1])
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out