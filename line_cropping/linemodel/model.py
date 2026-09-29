"""Phase 7 network: U-Net with a ResNet-34 encoder, outputs at stride 4.

Encoder (torchvision resnet34, ImageNet weights): stem stride 2 (64 ch), layer1 stride 4 (64), layer2 stride 8 (128),
layer3 stride 16 (256), layer4 stride 32 (512). Decoder: upsample x2, concatenate the skip of that resolution, two
3x3 conv+BN+ReLU; three steps back to stride 4; 1x1 head -> 4 channels (kernel logit, band logit, sin2t, cos2t).
"""
import torch, torch.nn as nn, torch.nn.functional as F, torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LineUNet(nn.Module):
    def __init__(self, pretrained=True, out_ch=4):
        super().__init__(); r = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu); self.pool = r.maxpool; self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d3 = block(512 + 256, 256); self.d2 = block(256 + 128, 128); self.d1 = block(128 + 64, 64); self.head = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        s = self.stem(x); f1 = self.l1(self.pool(s)); f2 = self.l2(f1); f3 = self.l3(f2); f4 = self.l4(f3)
        u = self.d3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False), f3], 1))
        u = self.d2(torch.cat([F.interpolate(u, size=f2.shape[-2:], mode='bilinear', align_corners=False), f2], 1))
        u = self.d1(torch.cat([F.interpolate(u, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1], 1))
        return self.head(u)


def line_loss(out, T):
    """out: (B,4,h,w) logits/values; T: (B,5,h,w) targets. Returns total, dict of parts."""
    kern_t, band_t, ign = T[:, 0], T[:, 1], T[:, 4]; valid = 1.0 - ign
    def bce_dice(logit, tgt):
        bce = (F.binary_cross_entropy_with_logits(logit, tgt, reduction='none') * valid).sum() / valid.sum().clamp(min=1)
        p = torch.sigmoid(logit) * valid; t = tgt * valid; dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return bce + dice
    lk, lb = bce_dice(out[:, 0], kern_t), bce_dice(out[:, 1], band_t)
    m = band_t * valid; lo = ((out[:, 2] - T[:, 2]).abs() + (out[:, 3] - T[:, 3]).abs()) * m; lo = lo.sum() / m.sum().clamp(min=1)
    total = lk + lb + 0.5 * lo
    return total, dict(kernel=float(lk), band=float(lb), orient=float(lo))
