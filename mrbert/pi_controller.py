"""
PI controller for targeting a specific deletion ratio (paper Section 3.2).
Updates alpha dynamically: alpha_{t+1} = clamp(kp * p_{t+1} + ki * i_{t+1}).
"""
import torch


class PIController:
    """Proportional-Integral controller for gate regularizer weight alpha."""

    def __init__(self, target_ratio: float, kp: float = 0.5, ki: float = 1e-5, gamma: float = 0.9):
        self.target_ratio = target_ratio
        self.kp = kp
        self.ki = ki
        self.gamma = gamma
        self.p = 0.0
        self.i = 0.0

    def step(self, gate: torch.Tensor, gate_k: float = -30.0) -> float:
        """Update alpha from current batch gate. gate shape: (batch, seq_len), values in [gate_k, 0]."""
        with torch.no_grad():
            threshold = gate_k * 0.5  # paper: k/2
            deleted = (gate < threshold).float()
            current_ratio = deleted.mean().item()
            current_ratio = max(0.0, min(1.0, current_ratio))
        error = self.target_ratio - current_ratio
        self.p = self.gamma * self.p + (1 - self.gamma) * error
        self.i = self.i + error
        alpha = max(0.0, self.kp * self.p + self.ki * self.i)
        return alpha
