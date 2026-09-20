"""Actual PyTorch graph neural network layers used by train_real_gnn.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNRegressor(nn.Module):
    """Two-layer spectral GCN: normalized road adjacency is used in both message-passing layers."""
    def __init__(self, in_dim=9, hidden=32):
        super().__init__(); self.w1=nn.Linear(in_dim,hidden); self.w2=nn.Linear(hidden,1)
    def forward(self,A,X):
        H=F.relu(A @ self.w1(X)); return (A @ self.w2(H)).squeeze(-1)

class TopologyAdaptiveGNN(nn.Module):
    """GAT-style GNN: learned attention reweights messages on the physical road graph."""
    def __init__(self, in_dim=9, hidden=32):
        super().__init__(); self.lin=nn.Linear(in_dim,hidden,bias=False); self.a=nn.Linear(2*hidden,1,bias=False); self.out=nn.Linear(hidden,1)
    def forward(self,A,X,return_attention=False):
        H=F.relu(self.lin(X)); n=H.shape[0]
        hi=H.unsqueeze(1).expand(n,n,-1); hj=H.unsqueeze(0).expand(n,n,-1)
        e=F.leaky_relu(self.a(torch.cat([hi,hj],-1)).squeeze(-1),0.2).masked_fill(A<=0,-1e9)
        alpha=F.softmax(e,dim=1); y=self.out(F.relu(alpha@H)).squeeze(-1)
        return (y,alpha) if return_attention else y
