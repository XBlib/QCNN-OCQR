import torch.nn as nn
import torch
import torch.nn.functional as F

import Circuit

import pennylane as qml

class RegionShapeQuantumKernel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        qnode = qml.QNode(
            Circuit.circuit_RegionShape_Ry_4qubits,
            dev,
            diff_method="best",
        )
        self.quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (8,)})

    def forward(self, inputs):
        measurements = self.quantum_layer(inputs)
        z = measurements[..., :4]
        zz = measurements[..., 4:]

        products = torch.stack(
            (
                z[..., 0] * z[..., 1],
                z[..., 2] * z[..., 3],
                z[..., 0] * z[..., 2],
                z[..., 1] * z[..., 3],
            ),
            dim=-1,
        )
        connected_correlations = zz - products
        descriptor = torch.cat((z, connected_correlations), dim=-1)
        return descriptor

class DirectionalEdgeQuantumKernel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        qnode = qml.QNode(
            Circuit.circuit_DirectionalEdge_Ry_4qubits,
            dev,
            diff_method="best",
        )
        self.quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (10,)})

    def forward(self, inputs):
        measurements = self.quantum_layer(inputs)
        z = measurements[..., :4]
        zz = measurements[..., 4:]
        pairs = ((0, 1), (2, 3), (0, 2), (1, 3))
        products = torch.stack([z[..., i] * z[..., j] for i, j in pairs], dim=-1)
        return torch.cat((z, zz - products), dim=-1)

class TextureQuantumKernel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        qnode = qml.QNode(
            Circuit.circuit_Texture_Ry_4qubits,
            dev,
            diff_method="best",
        )
        self.quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (12,)})

    def forward(self, inputs):
        measurements = self.quantum_layer(inputs)
        z = measurements[..., :4]
        zz = measurements[..., 4:]

        pair_indices = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
        products = torch.stack(
            [z[..., i] * z[..., j] for i, j in pair_indices],
            dim=-1,
        )
        connected_correlations = zz - products
        return torch.cat((z, connected_correlations), dim=-1)

class CornerCurvatureQuantumKernel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        qnode = qml.QNode(
            Circuit.circuit_CornerCurvature_4qubits,
            dev,
            diff_method="best",
        )
        self.quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (12,)})

    def forward(self, inputs):
        measurements = self.quantum_layer(inputs)
        z, zz = measurements[..., :4], measurements[..., 4:]
        pairs = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
        products = torch.stack([z[..., i] * z[..., j] for i, j in pairs], dim=-1)
        return torch.cat((z, zz - products), dim=-1)

class RGBCrossChannelQuantumKernel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        qnode = qml.QNode(
            Circuit.circuit_RGBCrossChannel_4qubits,
            dev,
            diff_method="best",
        )
        self.quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (10,)})

    def forward(self, rgb_patch):
        if rgb_patch.ndim != 2 or rgb_patch.shape[-1] != 12:
            raise ValueError(
                "RGB cross-channel QNode requires [batch, 12] input "
                "with R/G/B groups of four pixels each"
            )
        measurements = self.quantum_layer(rgb_patch)
        z = measurements[..., :4]
        zz = measurements[..., 4:]
        pairs = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
        products = torch.stack([z[..., i] * z[..., j] for i, j in pairs], dim=-1)
        return torch.cat((z, zz - products), dim=-1)

class OverlapConsistentQuantumReadout(nn.Module):
    # 每个量子比特对应水平、垂直和对角三个测量配对。
    PER_Q = ((0, 2, 4), (0, 3, 5), (1, 2, 5), (1, 3, 4))
    # 将块内四个位置分别对齐到原图坐标。
    ALIGN_PADS = ((0, 1, 0, 1), (1, 0, 0, 1),
                  (0, 1, 1, 0), (1, 0, 1, 0))

    def __init__(self, colors, measured_pairs, latent_dim=8):
        super().__init__()
        if colors not in (1, 3) or measured_pairs not in (4, 6):
            raise ValueError('OCQR supports RGB-separated 3 or joint-RGB 1 input, 4/6 ZZ pairs')
        self.colors = colors
        self.measured_pairs = measured_pairs
        self.latent_dim = latent_dim
        # 四个量子比特共用同一组局部嵌入参数。
        self.local_embedding = nn.Conv2d(colors * 5, latent_dim, 1)
        self.signed_role_weights = nn.Parameter(torch.randn(4, latent_dim) * .05)
        self.merge = nn.Sequential(
            nn.Conv2d(3 * latent_dim, latent_dim, 1, bias=False),
            nn.BatchNorm2d(latent_dim), nn.GELU(),
        )

    def patch_embeddings(self, descriptors):
        if descriptors.ndim != 4 or descriptors.shape[1] != self.colors * (4 + self.measured_pairs):
            raise ValueError('Quantum descriptor channel count mismatch')
        b, _, h, w = descriptors.shape
        d = descriptors.reshape(b, self.colors, 4 + self.measured_pairs, h, w)
        z = d[:, :, :4]
        cc = d[:, :, 4:]
        missing = z.new_zeros((b, self.colors, h, w))
        flag = z.new_full((b, self.colors, h, w), 1.0 if self.measured_pairs == 6 else 0.0)
        pieces = []
        for q, (h_pair, v_pair, diag_pair) in enumerate(self.PER_Q):
            h_cc, v_cc = cc[:, :, h_pair], cc[:, :, v_pair]
            diag_cc = cc[:, :, diag_pair] if diag_pair < self.measured_pairs else missing
            pieces.append(torch.stack((z[:, :, q], h_cc, v_cc, diag_cc, flag), dim=2)
                          .reshape(b, self.colors * 5, h, w))
        local = torch.stack(pieces, dim=1).reshape(b*4, self.colors*5, h, w)
        return self.local_embedding(local).reshape(b, 4, self.latent_dim, h, w)

    def align_and_decompose(self, window_features):
        if window_features.ndim != 5 or window_features.shape[1:3] != (4, self.latent_dim):
            raise ValueError('Expected [B,4,d,H,W] patch-position embeddings')
        b, _, d, h, w = window_features.shape
        aligned = torch.stack([F.pad(window_features[:, q], pad)
                               for q, pad in enumerate(self.ALIGN_PADS)], dim=1)
        valid = torch.stack([F.pad(window_features.new_ones((1, 1, h, w)), pad)
                             for pad in self.ALIGN_PADS], dim=1)
        # 按重叠次数归一化，并保留窗口间的差异。
        counts = valid.sum(dim=1)
        mean = aligned.sum(dim=1) / counts
        residual = (aligned - mean.unsqueeze(1)) * valid
        energy = (residual.square().sum(dim=1) / counts + 1e-12).sqrt()
        contrast = (residual * self.signed_role_weights[None, :, :, None, None]).sum(dim=1) / counts
        return aligned, valid, mean, contrast, energy

    def forward(self, descriptors):
        patches = self.patch_embeddings(descriptors)
        _, _, mean, contrast, energy = self.align_and_decompose(patches)
        return self.merge(torch.cat((mean, contrast, energy), dim=1))

class LightweightSpatialEncoder(nn.Module):
    def __init__(self, channels=40):
        super().__init__()
        def pathway(dilation):
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation,
                          dilation=dilation, groups=channels, bias=False),
                nn.BatchNorm2d(channels), nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.local = pathway(1)
        self.context = pathway(2)

    def forward(self, x):
        return F.gelu(x + .5 * self.local(x) + .5 * self.context(x))

class OCQR_Conv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, kernel_size=2,
                 stride=1):
        super().__init__()
        if in_channels != 3 or out_channels != 5 or kernel_size != 2 or stride != 1:
            raise ValueError('OCQR formal model requires RGB, five kernels, 2x2 patches, stride=1')
        self.dev = qml.device('default.qubit', wires=4)
        kernel_types = (RegionShapeQuantumKernel, DirectionalEdgeQuantumKernel,
                        TextureQuantumKernel, CornerCurvatureQuantumKernel,
                        RGBCrossChannelQuantumKernel)
        self.kernels = nn.ModuleList([
            nn.ModuleList([typ(self.dev) for _ in range(1 if i == 4 else 3)])
            for i, typ in enumerate(kernel_types)
        ])
        self.readouts = nn.ModuleList([
            OverlapConsistentQuantumReadout(colors=3 if i < 4 else 1,
                                            measured_pairs=4 if i < 2 else 6,
                                            latent_dim=8)
            for i in range(5)
        ])
        self.spatial = LightweightSpatialEncoder(40)

    def quantum_feature_maps(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3 or min(inputs.shape[-2:]) < 2:
            raise ValueError('Expected [B,3,H,W] with H,W >= 2')
        batch, _, height, width = inputs.shape
        response = [[] for _ in range(5)]
        for row in range(height - 1):
            for col in range(width - 1):
                patch = inputs[:, :, row:row+2, col:col+2].reshape(batch, 3, 4)
                for k in range(5):
                    if k == 4:
                        descriptor = self.kernels[k][0](patch.reshape(batch, 12)).reshape(batch, -1)
                    else:
                        descriptor = torch.cat([
                            self.kernels[k][color](patch[:, color:color+1, :]).reshape(batch, -1)
                            for color in range(3)
                        ], dim=1)
                    response[k].append(descriptor)
        return [torch.stack(response[k], dim=-1)
                .reshape(batch, 3 * (8 if k < 2 else 10) if k < 4 else 10,
                         height - 1, width - 1)
                for k in range(5)]

    def forward(self, inputs):
        maps = self.quantum_feature_maps(inputs)
        aligned = [head(desc) for head, desc in zip(self.readouts, maps)]
        return self.spatial(torch.cat(aligned, dim=1))

class OCQRClassifierHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = nn.BatchNorm2d(40)
        self.avg = nn.AdaptiveAvgPool2d((2, 2))
        self.maximum = nn.AdaptiveMaxPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Linear(40*2*2*2, 64), nn.ReLU(),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.norm(x)
        pooled = torch.cat((self.avg(x).flatten(1), self.maximum(x).flatten(1)), dim=1)
        return self.classifier(pooled)
