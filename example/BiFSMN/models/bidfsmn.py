import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BiDfsmnLayer(nn.Layer):
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
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, backbone_memory_size),
            nn.Dropout(dropout)
        )
        self.memory = nn.Conv1D(
            in_channels=backbone_memory_size,
            out_channels=backbone_memory_size,
            kernel_size=left_kernel_size + right_kernel_size + 1,
            padding=0,
            stride=1,
            dilation=dilation,
            groups=backbone_memory_size
        )

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        residual = input_feat
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out
        fc_output = self.fc_trans(memory_out.transpose([0, 2, 1]))  # (B, T, N)
        output = fc_output.transpose([0, 2, 1]) + residual  # (B, N, T)
        return output


class BiDfsmnLayerBN(nn.Layer):
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
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1D(hidden_size, backbone_memory_size, 1),
            nn.BatchNorm1D(backbone_memory_size),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        self.memory = nn.Sequential(
            nn.Conv1D(
                backbone_memory_size,
                backbone_memory_size,
                kernel_size=left_kernel_size + right_kernel_size + 1,
                padding=0,
                stride=1,
                dilation=dilation,
                groups=backbone_memory_size
            ),
            nn.BatchNorm1D(backbone_memory_size),
            nn.PReLU()
        )

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        residual = input_feat
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        output = fc_output + residual  # (B, N, T)
        return output


class BiDfsmnModel(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.0,
                 dfsmn_with_bn=True,
                 distill=False,
                 **kwargs):
        super().__init__()
        self.front_end = nn.Sequential(
            nn.Conv2D(
                in_channels, frondend_channels,
                kernel_size=[frondend_kernel_size, frondend_kernel_size],
                stride=(2, 2),
                padding=(frondend_kernel_size // 2, frondend_kernel_size // 2)
            ),
            nn.BatchNorm2D(frondend_channels),
            nn.PReLU(),
            nn.Conv2D(
                frondend_channels, 2 * frondend_channels,
                kernel_size=[frondend_kernel_size, frondend_kernel_size],
                stride=(2, 2),
                padding=(frondend_kernel_size // 2, frondend_kernel_size // 2)
            ),
            nn.BatchNorm2D(2 * frondend_channels),
            nn.PReLU()
        )
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(
            nn.Linear(2 * frondend_channels * self.n_mels // 4, backbone_memory_size),
            nn.PReLU(),
        )
        backbone = []
        for idx in range(num_layer):
            if dfsmn_with_bn:
                backbone.append(
                    BiDfsmnLayerBN(hidden_size, backbone_memory_size,
                                    left_kernel_size, right_kernel_size, dilation, dropout)
                )
            else:
                backbone.append(
                    BiDfsmnLayer(hidden_size, backbone_memory_size,
                                  left_kernel_size, right_kernel_size, dilation, dropout)
                )
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_memory_size * 32 // 4, num_classes)
        )
        self.distill = distill

    def forward(self, input_feat):
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = paddle.flatten(out, start_axis=1, stop_axis=-2)  # B, T, N1
        out = self.fc1(out).transpose([0, 2, 1])  # B, N, T
        features = []
        for layer in self.backbone:
            out = layer(out)
            features.append(out)
        out = paddle.flatten(out, start_axis=1)  # Flatten and reshape to batch size
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out
        

class BiDfsmnLayerBN_thinnable(nn.Layer):
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
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1D(hidden_size, backbone_memory_size, 1),
        )
        self.bn0 = nn.BatchNorm1D(backbone_memory_size)
        self.act0 = nn.PReLU()
        self.bn1 = nn.BatchNorm1D(backbone_memory_size)
        self.act1 = nn.PReLU()
        self.bn2 = nn.BatchNorm1D(backbone_memory_size)
        self.act2 = nn.PReLU()
        self.bn3 = nn.BatchNorm1D(backbone_memory_size)
        self.act3 = nn.PReLU()
        self.memory = nn.Sequential(
            nn.Conv1D(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1D(backbone_memory_size),
            nn.PReLU(),
        )

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat, opt):
        residual = input_feat
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B, N, T + (l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        if opt == 0:
            fc_output = self.bn0(fc_output)
            fc_output = self.act0(fc_output)
        elif opt == 1:
            fc_output = self.bn1(fc_output)
            fc_output = self.act1(fc_output)
        elif opt == 2:
            fc_output = self.bn2(fc_output)
            fc_output = self.act2(fc_output)
        elif opt == 3:
            fc_output = self.bn3(fc_output)
            fc_output = self.act3(fc_output)
        else:
            raise Exception(f'opt should be in [0, 1, 2, 3], but got {opt}')
        
        output = fc_output + residual  # (B, N, T)
        return output


class BiDfsmnModel_thinnable(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 dropout=0.0,
                 dfsmn_with_bn=True,
                 thin_n=3,
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
            nn.PReLU(),
            nn.Conv2D(frondend_channels,
                      out_channels=2 * frondend_channels,
                      kernel_size=[frondend_kernel_size, frondend_kernel_size],
                      stride=(2, 2),
                      padding=(frondend_kernel_size // 2, frondend_kernel_size // 2)),
            nn.BatchNorm2D(2 * frondend_channels),
            nn.PReLU()
        )
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(
            nn.Linear(2 * frondend_channels * self.n_mels // 4, backbone_memory_size),
            nn.PReLU(),
        )
        self.backbone = nn.LayerList([
            BiDfsmnLayerBN_thinnable(hidden_size, backbone_memory_size,
                                      left_kernel_size, right_kernel_size, dilation, dropout)
            for _ in range(num_layer)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_memory_size * 32 // 4, num_classes),
        )
        self.distill = distill
        self.thin_n = thin_n

    def forward(self, input_feat, opt):
        batch = input_feat.shape[0]
        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.reshape([batch, -1, out.shape[3]]).transpose([0, 2, 1])  # B, T, N1
        out = self.fc1(out).transpose([0, 2, 1])  # B, N, T
        features = []
        if opt == 0:
            for idx in range(8):
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 1:
            for idx in [1, 3, 5, 7]:
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 2:
            for idx in [3, 7]:
                out = self.backbone[idx](out, opt)
                features.append(out)
        elif opt == 3:
            for idx in [7]:
                out = self.backbone[idx](out, opt)
                features.append(out)

        out = out.reshape([batch, -1])
        out = self.classifier(out)
        if self.distill:
            return out, features
        else:
            return out


class DfsmnLayerBN_pre(nn.Layer):
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
        )
        self.bn0 = nn.BatchNorm1D(backbone_memory_size)
        self.act0 = nn.PReLU()
        self.memory = nn.Sequential(
            nn.Conv1D(backbone_memory_size,
                      backbone_memory_size,
                      kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            nn.BatchNorm1D(backbone_memory_size),
            nn.PReLU(),
        )

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        residual = input_feat
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B, N, T + (l+r)*d)
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)

        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        fc_output = self.bn0(fc_output)
        fc_output = self.act0(fc_output)
        output = fc_output + residual  # (B, N, T)
        return output


class DfsmnModel_pre(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_mels=32,
                 num_layer=8,
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
            nn.Linear(2 * frondend_channels * self.n_mels // 4, backbone_memory_size),
            nn.ReLU(),
        )
        self.backbone = nn.LayerList([
            DfsmnLayerBN_pre(hidden_size, backbone_memory_size,
                             left_kernel_size, right_kernel_size, dilation, dropout)
            for _ in range(num_layer)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_memory_size * self.n_mels // 4, num_classes),
        )
        self.distill = distill

    def forward(self, input_feat):
        batch = input_feat.shape[0]
        # print(f"input_feat.shape: {input_feat.shape}")
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