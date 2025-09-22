import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, in_channels, channels, num_layers=4, kernel_size=3, dilations=[1,2,4,8]):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d = dilations[i]
            layers.append(TemporalBlock(
                in_channels if i == 0 else channels,
                channels,
                kernel_size,
                dilation=d,
                padding=(kernel_size-1)*d//2
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]
        return self.network(x)

class MultiScaleRegionAttention(nn.Module):
    def __init__(self, max_region_size=3):
        super().__init__()
        self.max_region_size = max_region_size

    def forward(self, x):
        # x: [B, C, T]
        region_embeds = []
        for region in range(1, self.max_region_size+1):
            key = F.max_pool1d(x, kernel_size=region, stride=1, padding=region//2)
            value = F.avg_pool1d(x, kernel_size=region, stride=1, padding=region//2) * region
            region_embed = key * value
            region_embeds.append(region_embed)
        # Fix: trim all to the minimum time length
        min_len = min(embed.shape[2] for embed in region_embeds)
        region_embeds = [embed[:, :, :min_len] for embed in region_embeds]
        out = sum(region_embeds) / len(region_embeds)
        # Global average pooling over time
        return out.mean(dim=2)  # [B, C]

class StandardAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [B, S, D]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (x.size(-1) ** 0.5), dim=-1)
        return (attn @ V).sum(dim=1)  # [B, D]

# Remove MultiScaleRegionAttention, StandardAttention, and all 'return_attention' logic from HATCNInterpretable. Only keep the core model logic for classification and SHAP analysis.
class HATCNInterpretable(nn.Module):
    def __init__(self, input_dim=83, tcn_channels=32, tcn_layers=4, kernel_size=3, dilations=[1,2,4,8], max_region_size=3, hidden_dim=128, num_classes=2):
        super().__init__()
        # Stage 1: Intra-sentence TCN + multi-scale region attention
        self.tcn1 = TCN(input_dim, tcn_channels, num_layers=tcn_layers, kernel_size=kernel_size, dilations=dilations)
        self.region_attn = MultiScaleRegionAttention(max_region_size=max_region_size)
        # Stage 2: Inter-sentence TCN + standard attention
        self.tcn2 = TCN(tcn_channels, tcn_channels, num_layers=2, kernel_size=kernel_size, dilations=[1,2])
        self.attn = StandardAttention(tcn_channels)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(tcn_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [B, T, F] or [B, S, T, F]
        if x.dim() == 3:
            # Single sentence per sample: [B, T, F]
            x = x.transpose(1,2)  # [B, F, T]
            tcn_out = self.tcn1(x)  # [B, C, T]
            sent_embed = self.region_attn(tcn_out)  # [B, C]
            # Add sentence dim for compatibility
            sent_embed = sent_embed.unsqueeze(1)  # [B, 1, C]
        elif x.dim() == 4:
            # Multiple sentences: [B, S, T, F]
            B, S, T, F = x.shape
            x = x.view(B*S, T, F).transpose(1,2)  # [B*S, F, T]
            tcn_out = self.tcn1(x)  # [B*S, C, T]
            sent_embed = self.region_attn(tcn_out)  # [B*S, C]
            sent_embed = sent_embed.view(B, S, -1)  # [B, S, C]
        else:
            raise ValueError('Input must be 3D or 4D tensor')
        # Stage 2: Inter-sentence TCN + attention
        tcn2_in = sent_embed.transpose(1,2)  # [B, C, S]
        tcn2_out = self.tcn2(tcn2_in)  # [B, C, S]
        tcn2_out = tcn2_out.transpose(1,2)  # [B, S, C]
        seg_embed = self.attn(tcn2_out)  # [B, C]
        seg_embed = self.dropout(seg_embed)
        logits = self.fc(seg_embed)
        return logits

# Backward compatibility - alias for existing code
HATCN = HATCNInterpretable 