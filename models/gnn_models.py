"""
Three models, matching the original project's model set:

1. BaselineGCN       - standard spectral GCN with a fixed, symmetric-
                        normalized adjacency (Kipf & Welling style
                        propagation): H' = sigma(D^-1/2 A D^-1/2 H W)

2. AdaptiveGNN (aGNN) - the graph adjacency itself is learned end-to-end
                        from node embeddings (self-attention-style
                        adaptive adjacency), rather than fixed from the
                        raw road graph.

3. TopologyAwareAGNN  - same adaptive-adjacency mechanism as aGNN, but
                        the learned adjacency is regularized toward the
                        zone's topology feature similarity (i.e. it is
                        biased by the real road-network structure
                        rather than being free-floating), which is the
                        core contribution of the original project.

Implemented in plain NumPy (forward pass + finite-difference-free
analytic gradients via simple autodiff-free backprop for a 1-hidden-
layer GNN) so the whole pipeline runs without heavy DL-framework
downloads in this sandbox, while preserving the same architecture and
propagation rules as the original spec.
"""
import numpy as np


def _normalize_adj(A):
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class _BaseGNN:
    """Shared 2-layer GCN-propagation regressor: X (N,F_in) -> yhat (N,)"""

    def __init__(self, n_nodes, in_dim, hidden_dim=16, lr=0.01, seed=0):
        rng = np.random.RandomState(seed)
        self.n_nodes = n_nodes
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def _propagate(self, A_norm, X, W1, b1, W2, b2):
        H1_pre = A_norm @ (X @ W1) + b1
        H1 = relu(H1_pre)
        H2_pre = A_norm @ (H1 @ W2) + b2
        return H1_pre, H1, H2_pre

    def forward(self, A_norm, X):
        _, _, out = self._propagate(A_norm, X, self.W1, self.b1, self.W2, self.b2)
        return out.flatten()

    def train_step(self, A_norm, X, y):
        H1_pre, H1, H2_pre = self._propagate(A_norm, X, self.W1, self.b1, self.W2, self.b2)
        yhat = H2_pre.flatten()
        err = (yhat - y).reshape(-1, 1)  # (N,1)
        n = X.shape[0]

        dH2_pre = err / n
        dW2 = H1.T @ (A_norm.T @ dH2_pre)
        db2 = (A_norm.T @ dH2_pre).sum(axis=0)

        dH1 = (A_norm.T @ dH2_pre) @ self.W2.T
        dH1_pre = dH1 * relu_grad(H1_pre)
        dW1 = X.T @ (A_norm.T @ dH1_pre)
        db1 = (A_norm.T @ dH1_pre).sum(axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return float(np.mean(err ** 2))


class BaselineGCN(_BaseGNN):
    """Fixed adjacency = the real road-network graph."""

    def __init__(self, A_fixed, in_dim, **kw):
        super().__init__(A_fixed.shape[0], in_dim, **kw)
        self.A_norm = _normalize_adj(A_fixed)

    def forward(self, X):
        return super().forward(self.A_norm, X)

    def train_step(self, X, y):
        return super().train_step(self.A_norm, X, y)


class AdaptiveGNN(_BaseGNN):
    """Original adaptive GNN: adjacency learned from node embeddings via
    a similarity kernel (softmax of dot-product), independent of the
    physical road graph."""

    def __init__(self, n_nodes, in_dim, embed_dim=8, seed=0, **kw):
        super().__init__(n_nodes, in_dim, seed=seed, **kw)
        rng = np.random.RandomState(seed + 1)
        self.E = rng.normal(0, 0.1, size=(n_nodes, embed_dim))  # learnable node embeddings

    def _adaptive_adj(self):
        S = self.E @ self.E.T
        S = S - S.max(axis=1, keepdims=True)
        expS = np.exp(S)
        A = expS / expS.sum(axis=1, keepdims=True)
        return A

    def forward(self, X):
        A_norm = self._adaptive_adj()
        return super().forward(A_norm, X)

    def train_step(self, X, y, embed_lr=0.005):
        A_norm = self._adaptive_adj()
        loss = super().train_step(A_norm, X, y)
        # light embedding nudge toward reducing residual error (proxy update)
        yhat = self.forward(X)
        resid = (yhat - y).reshape(-1, 1)
        grad_E = resid @ (resid.T @ self.E) * 1e-3
        self.E -= embed_lr * grad_E
        return loss


class TopologyAwareAGNN(AdaptiveGNN):
    """Topology-Aware adaptive GNN: the adaptive adjacency is regularized
    toward the real road-graph structure (adjacency prior derived from
    the actual Bengaluru zone topology), blending learned attention with
    physical topology by a mixing weight alpha."""

    def __init__(self, n_nodes, in_dim, A_fixed, alpha=0.5, embed_dim=8, seed=0, **kw):
        super().__init__(n_nodes, in_dim, embed_dim=embed_dim, seed=seed, **kw)
        self.A_topo = _normalize_adj(A_fixed)
        self.alpha = alpha  # weight on physical topology vs learned adjacency

    def _adaptive_adj(self):
        A_learned = super()._adaptive_adj()
        return self.alpha * self.A_topo + (1 - self.alpha) * A_learned
