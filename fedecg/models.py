import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(252,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,x,rr):
        return self.net(x.flatten(1))


class ECGCNN(nn.Module):
    def __init__(self, use_rr=False):
        super().__init__()
        self.use_rr=use_rr
        self.encoder=nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),nn.GroupNorm(4,16),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(4,32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.GroupNorm(8,64),nn.ReLU(),nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64+(4 if use_rr else 0),64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,x,rr):
        h=self.encoder(x)
        return self.head(torch.cat([h,rr],dim=1) if self.use_rr else h)


class MLPRR(nn.Module):
    def __init__(self):
        super().__init__()
        self.morphology=nn.Sequential(nn.Linear(252,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.timing=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(80,64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,x,rr):
        return self.head(torch.cat([self.morphology(x.flatten(1)),self.timing(rr)],dim=1))


class SpatialCNNRR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),nn.GroupNorm(4,16),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(4,32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),nn.MaxPool1d(2),
            nn.Flatten(),nn.Linear(32*31,64),nn.ReLU())
        self.timing=nn.Sequential(nn.Linear(4,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(80,64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,x,rr):
        return self.head(torch.cat([self.encoder(x),self.timing(rr)],dim=1))


def make_model(name):
    if name=='mlp': return MLP()
    if name=='mlp_rr': return MLPRR()
    if name=='cnn_spatial_rr': return SpatialCNNRR()
    if name in ['cnn','cnn_rr']: return ECGCNN(name=='cnn_rr')
    raise ValueError(name)


def class_weights(y,mode='weighted'):
    counts=torch.bincount(y,minlength=5).float()
    # Square-root inverse frequency caps the influence of the tiny Q class.
    w=(counts.sum()/counts.clamp_min(1)).clamp_max(100) if mode=='balanced' else torch.sqrt(counts.sum()/counts.clamp_min(1)).clamp_max(10)
    w[counts==0]=0
    return w/w[counts>0].mean()


def loss_fn(logits, y, kind, weights, gamma=2.):
    if kind=='ce': return F.cross_entropy(logits,y)
    if kind in ['weighted','balanced']: return F.cross_entropy(logits,y,weight=weights)
    if kind=='focal':
        # Native cross entropy avoids MPS gather's unsupported deterministic
        # scatter-reduce backward while giving exactly the target log probability.
        logp=-F.cross_entropy(logits,y,reduction='none')
        w=weights[y]
        return (-w*(1-logp.exp()).pow(gamma)*logp).sum()/w.sum().clamp_min(1e-8)
    raise ValueError(kind)
