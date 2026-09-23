"""Inference wrapper for the per-script sign classifiers trained by train_classifier.py."""
import json, os
import numpy as np, cv2, torch, torch.nn as nn, torchvision
from line_cropping.signs.train_classifier import letterbox, MEAN, STD, WEIGHTS


class SignClassifier:
    def __init__(self, script, device=None):
        d = os.path.join(WEIGHTS, script); m = json.load(open(os.path.join(d, 'metrics.json'), encoding='utf-8'))
        self.classes = json.load(open(os.path.join(d, 'classes.json'), encoding='utf-8')); self.index = {s: i for i, s in enumerate(self.classes)}
        self.size = m['size']; self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        net = torchvision.models.resnet34(weights=None); net.fc = nn.Linear(net.fc.in_features, len(self.classes))
        net.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu')); self.net = net.to(self.dev).eval()

    @torch.no_grad()
    def probs(self, bgr_crops, bs=256):
        """Softmax over the classifier's sign classes for each BGR crop -> array (N, n_classes)."""
        if not bgr_crops: return np.zeros((0, len(self.classes)), np.float32)
        X = np.stack([letterbox(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), self.size) for c in bgr_crops]); out = []
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).to(self.dev).float().div_(255); x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
            out.append(torch.softmax(self.net(x).float(), dim=1).cpu().numpy())
        return np.concatenate(out)
