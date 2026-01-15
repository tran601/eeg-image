import torch
import torch.nn as nn
import torch.nn.functional as F


class TextPathLoss(nn.Module):
    """
    Sinkhorn-based token-level alignment loss between EEG and text embeddings.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        n_iters: int = 50,
        sim_temperature: float = 0.1,
        gamma_pos: float = 0.02,
        barycentric_weight: float = 0.5,
        global_cosine_weight: float = 0.2,
        use_sinkhorn_divergence: bool = False,
        position_metric: str = "l1",
        reduce: str = "mean",
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.n_iters = int(n_iters)
        self.sim_temperature = float(sim_temperature)
        self.gamma_pos = float(gamma_pos)
        self.barycentric_weight = float(barycentric_weight)
        self.global_cosine_weight = float(global_cosine_weight)
        self.use_sinkhorn_divergence = bool(use_sinkhorn_divergence)
        self.position_metric = position_metric
        if reduce not in {"mean", "sum", "none"}:
            raise ValueError("reduce must be 'mean', 'sum', or 'none'")
        self.reduce = reduce

        # Cache position priors for reuse across batches
        self._pos_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, eeg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        X, Y = self._ensure_batch(eeg_embedding, text_embedding)
        Xn = F.normalize(X, dim=-1)
        Yn = F.normalize(Y, dim=-1)

        B, N, _ = Xn.shape

        S = torch.matmul(Xn, Yn.transpose(-1, -2))
        C_content = (1.0 - S).clamp_min(0.0) / self.sim_temperature

        if self.gamma_pos > 0.0:
            P = self._get_position_prior(N, device=X.device, dtype=X.dtype)
            C = C_content + self.gamma_pos * P
        else:
            C = C_content

        a = torch.full((B, N), 1.0 / N, device=X.device, dtype=X.dtype)
        b = torch.full((B, N), 1.0 / N, device=X.device, dtype=X.dtype)

        T, logT = self._sinkhorn_log_domain(C, a, b, eps=self.epsilon, n_iters=self.n_iters)

        if self.use_sinkhorn_divergence:
            L_main = self._sinkhorn_divergence(Xn, Yn, a, b, N)
        else:
            L_main = (T * C_content).sum(dim=(-1, -2))

        r = T.sum(dim=-1, keepdim=False).unsqueeze(-1)
        c = T.sum(dim=-2, keepdim=False).unsqueeze(-1)

        Y_bar = torch.matmul(T, Y) / (r + 1e-8)
        X_bar = torch.matmul(T.transpose(-1, -2), X) / (c + 1e-8)

        L_bary = (X - Y_bar).pow(2).mean(dim=(1, 2)) + (Y - X_bar).pow(2).mean(
            dim=(1, 2)
        )

        x_g = (X * r).sum(dim=1) / (r.sum(dim=1) + 1e-8)
        y_g = (Y * c).sum(dim=1) / (c.sum(dim=1) + 1e-8)
        x_g = F.normalize(x_g, dim=-1)
        y_g = F.normalize(y_g, dim=-1)
        L_cos_global = 1.0 - (x_g * y_g).sum(dim=-1)

        loss = (
            L_main
            + self.barycentric_weight * L_bary
            + self.global_cosine_weight * L_cos_global
        )
        return self._batch_reduce(loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_batch(X: torch.Tensor, Y: torch.Tensor):
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
        if X.dim() != 3 or Y.dim() != 3:
            raise ValueError("Inputs must be (B,N,D) or (N,D)")
        if X.shape[:2] != Y.shape[:2]:
            raise ValueError("X and Y must have the same (B,N)")
        return X, Y

    def _get_position_prior(self, N: int, device, dtype):
        key = (N, device, dtype, self.position_metric)
        if key in self._pos_cache:
            return self._pos_cache[key]

        idx = torch.arange(N, device=device, dtype=dtype)
        diff = (idx[:, None] - idx[None, :]) / max(N - 1, 1)
        if self.position_metric.lower() == "l2":
            P = diff.pow(2).abs()
        else:
            P = diff.abs()
        P = P.unsqueeze(0)
        self._pos_cache[key] = P
        return P

    def _sinkhorn_log_domain(self, C, a, b, eps: float, n_iters: int):
        B, N, _ = C.shape
        logK = -C / eps
        f = torch.zeros((B, N), device=C.device, dtype=C.dtype)
        g = torch.zeros((B, N), device=C.device, dtype=C.dtype)
        loga = torch.log(a + 1e-12)
        logb = torch.log(b + 1e-12)

        for _ in range(n_iters):
            f = loga - torch.logsumexp(logK + g.unsqueeze(1), dim=-1)
            g = logb - torch.logsumexp(logK.transpose(-1, -2) + f.unsqueeze(1), dim=-1)

        logT = logK + f.unsqueeze(-1) + g.unsqueeze(1)
        T = torch.exp(logT)
        return T, logT

    def _ot_primal_value(self, T, C, logT):
        cost = (T * C).sum(dim=(-1, -2))
        ent = (T * (logT - 1.0)).sum(dim=(-1, -2))
        return cost + self.epsilon * ent

    def _sinkhorn_divergence(self, Xn, Yn, a, b, N):
        S_xy = torch.matmul(Xn, Yn.transpose(-1, -2))
        C_xy_content = (1.0 - S_xy).clamp_min(0.0) / self.sim_temperature
        if self.gamma_pos > 0.0:
            P = self._get_position_prior(N, device=Xn.device, dtype=Xn.dtype)
            C_xy = C_xy_content + self.gamma_pos * P
        else:
            C_xy = C_xy_content

        T_xy, logT_xy = self._sinkhorn_log_domain(C_xy, a, b, eps=self.epsilon, n_iters=self.n_iters)
        ot_xy = self._ot_primal_value(T_xy, C_xy_content, logT_xy)

        S_xx = torch.matmul(Xn, Xn.transpose(-1, -2))
        C_xx_content = (1.0 - S_xx).clamp_min(0.0) / self.sim_temperature
        if self.gamma_pos > 0.0:
            P = self._get_position_prior(N, device=Xn.device, dtype=Xn.dtype)
            C_xx = C_xx_content + self.gamma_pos * P
        else:
            C_xx = C_xx_content

        T_xx, logT_xx = self._sinkhorn_log_domain(C_xx, a, a, eps=self.epsilon, n_iters=self.n_iters)
        ot_xx = self._ot_primal_value(T_xx, C_xx_content, logT_xx)

        S_yy = torch.matmul(Yn, Yn.transpose(-1, -2))
        C_yy_content = (1.0 - S_yy).clamp_min(0.0) / self.sim_temperature
        if self.gamma_pos > 0.0:
            P = self._get_position_prior(N, device=Yn.device, dtype=Yn.dtype)
            C_yy = C_yy_content + self.gamma_pos * P
        else:
            C_yy = C_yy_content

        T_yy, logT_yy = self._sinkhorn_log_domain(C_yy, b, b, eps=self.epsilon, n_iters=self.n_iters)
        ot_yy = self._ot_primal_value(T_yy, C_yy_content, logT_yy)

        sink_div = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
        return sink_div

    def _batch_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce == "none":
            return x
        if self.reduce == "mean":
            return x.mean()
        return x.sum()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 4, 77, 768
    eeg = torch.randn(B, N, D)
    txt = torch.randn(B, N, D)

    loss_fn = TextPathLoss(
        epsilon=0.1,
        n_iters=50,
        sim_temperature=0.1,
        gamma_pos=0.02,
        barycentric_weight=0.5,
        global_cosine_weight=0.2,
        use_sinkhorn_divergence=False,
        position_metric="l1",
        reduce="mean",
    )

    loss = loss_fn(eeg, txt)
    print(f"Sinkhorn OT loss = {loss.item():.6f}")
