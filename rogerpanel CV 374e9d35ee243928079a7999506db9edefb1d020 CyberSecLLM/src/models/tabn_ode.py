"""
Temporal Adaptive Batch Normalization Neural ODE (TA-BN-ODE).

Implements continuous-depth networks with time-dependent normalization
for security event sequence modeling, including multi-scale temporal
architecture, stability analysis components, and adaptive integration.

Reference: Section IV of the TA-BN-ODE paper (Theorem 1, Algorithm 1).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    odeint = None
    odeint_adjoint = None


class TemporalAdaptiveBatchNorm(nn.Module):
    """
    Temporal Adaptive Batch Normalization (TA-BN).

    Extends standard batch normalization to continuous time by
    parameterizing normalization statistics as functions of
    integration time t (Eq. 5 in the paper).

    Parameters gamma(t) and beta(t) are modeled through MLPs
    with periodic time features to capture cyclic patterns.
    """

    def __init__(self, num_features, mlp_hidden=64, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        time_input_dim = 3

        self.gamma_mlp = nn.Sequential(
            nn.Linear(time_input_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, num_features),
            nn.Softmax(dim=-1),
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(time_input_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, num_features),
        )

        self.omega = nn.Parameter(torch.tensor(2.0 * math.pi))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _time_features(self, t):
        if isinstance(t, (int, float)):
            t = torch.tensor([t], dtype=torch.float32, device=self.running_mean.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        return torch.stack([
            t,
            torch.sin(self.omega * t),
            torch.cos(self.omega * t),
        ], dim=-1)

    def forward(self, x, t):
        time_feat = self._time_features(t)

        gamma = self.gamma_mlp(time_feat)
        beta = self.beta_mlp(time_feat)

        if gamma.dim() < x.dim():
            for _ in range(x.dim() - gamma.dim()):
                gamma = gamma.unsqueeze(0)
                beta = beta.unsqueeze(0)

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return gamma * x_norm + beta


class TABNODEFunc(nn.Module):
    """
    ODE function with Temporal Adaptive Batch Normalization.

    Defines the dynamics dh/dt = f_theta(h, t) with TA-BN layers
    enabling stable stacking of multiple continuous blocks (Eq. 4).
    """

    def __init__(self, hidden_dim, mlp_hidden=64):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn1 = TemporalAdaptiveBatchNorm(hidden_dim, mlp_hidden)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.tabn2 = TemporalAdaptiveBatchNorm(hidden_dim, mlp_hidden)
        self.activation = nn.ELU()

    def forward(self, t, h):
        out = self.linear1(h)
        out = self.tabn1(out, t)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.tabn2(out, t)
        out = self.activation(out)
        return out


class MultiScaleODEFunc(nn.Module):
    """
    Multi-scale temporal ODE function (Eq. 7).

    Captures attack patterns across eight orders of magnitude
    through parallel ODE branches with learned time constants:
    dh/dt = sum_s alpha_s * f_{theta_s}(t / tau_s)
    """

    def __init__(
        self,
        hidden_dim,
        time_constants=None,
        mlp_hidden=64,
    ):
        super().__init__()
        if time_constants is None:
            time_constants = [1e-6, 1e-3, 1.0, 3600.0]

        self.time_constants = time_constants
        n_scales = len(time_constants)

        self.scale_funcs = nn.ModuleList([
            TABNODEFunc(hidden_dim, mlp_hidden) for _ in range(n_scales)
        ])

        self.attention_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

    def forward(self, t, h):
        alpha = F.softmax(self.attention_weights, dim=0)
        result = torch.zeros_like(h)
        for i, (tau, func) in enumerate(
            zip(self.time_constants, self.scale_funcs)
        ):
            scaled_t = t / tau if isinstance(t, (int, float)) else t / tau
            result = result + alpha[i] * func(scaled_t, h)
        return result


class TABNODEBlock(nn.Module):
    """
    Single TA-BN-ODE block implementing continuous-depth integration.

    Integrates the ODE from t0 to t1 using adaptive step-size
    Dormand-Prince (dopri5) method with adjoint backpropagation
    for O(1) memory complexity.
    """

    def __init__(
        self,
        hidden_dim,
        time_constants=None,
        solver="dopri5",
        rtol=1e-3,
        atol=1e-4,
        use_adjoint=True,
        mlp_hidden=64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        self.ode_func = MultiScaleODEFunc(
            hidden_dim, time_constants, mlp_hidden
        )

    def forward(self, h0, t_span):
        if odeint is None:
            return self._euler_forward(h0, t_span)

        integrator = odeint_adjoint if self.use_adjoint else odeint
        solution = integrator(
            self.ode_func,
            h0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        return solution[-1]

    def _euler_forward(self, h0, t_span, n_steps=10):
        """Fallback Euler integration when torchdiffeq is unavailable."""
        dt = (t_span[-1] - t_span[0]) / n_steps
        h = h0
        t = t_span[0]
        for _ in range(n_steps):
            h = h + dt * self.ode_func(t, h)
            t = t + dt
        return h


class EventUpdateNetwork(nn.Module):
    """
    Event-driven state update network.

    Updates the continuous state when discrete events occur:
    h(t_i^+) = h(t_i^-) + Update(x_i)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.update_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_features):
        return self.update_net(event_features)


class TABN_ODE(nn.Module):
    """
    Complete TA-BN-ODE module implementing Algorithm 1.

    Forward pass:
    1. Initialize h(t_0) = Encoder(x_0)
    2. For each event i:
       a. Integrate: h(t_i^-) = ODESolve(f_theta, h(t_{i-1}), [t_{i-1}, t_i])
       b. Update:    h(t_i) = h(t_i^-) + Update(x_i)
    3. Return continuous states {h(t_i)}
    """

    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        num_blocks=2,
        time_constants=None,
        solver="dopri5",
        rtol=1e-3,
        atol=1e-4,
        use_adjoint=True,
        mlp_hidden=64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.ode_blocks = nn.ModuleList([
            TABNODEBlock(
                hidden_dim, time_constants, solver, rtol, atol,
                use_adjoint, mlp_hidden,
            )
            for _ in range(num_blocks)
        ])

        self.event_update = EventUpdateNetwork(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, event_features, timestamps):
        """
        Forward pass implementing Algorithm 1.

        Args:
            event_features: (batch, seq_len, input_dim) event feature sequences
            timestamps: (batch, seq_len) event timestamps

        Returns:
            states: (batch, seq_len, hidden_dim) continuous states at each event
        """
        batch_size, seq_len, _ = event_features.shape
        h = self.encoder(event_features[:, 0])
        states = [h]

        for i in range(1, seq_len):
            t_span = torch.tensor(
                [timestamps[:, i - 1].mean().item(), timestamps[:, i].mean().item()],
                device=event_features.device,
            )
            if t_span[1] <= t_span[0]:
                t_span[1] = t_span[0] + 1e-6

            for block in self.ode_blocks:
                h = block(h, t_span)

            update = self.event_update(event_features[:, i])
            h = h + update
            states.append(h)

        states = torch.stack(states, dim=1)
        return self.output_proj(states)

    def compute_stability_regularization(self):
        """
        Compute regularization term enforcing TA-BN parameter bounds
        for adjoint gradient stability (Theorem 1).
        """
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.ode_blocks:
            for tabn in [
                block.ode_func.scale_funcs[j].tabn1
                for j in range(len(block.ode_func.scale_funcs))
            ] + [
                block.ode_func.scale_funcs[j].tabn2
                for j in range(len(block.ode_func.scale_funcs))
            ]:
                for p in tabn.gamma_mlp.parameters():
                    reg = reg + p.pow(2).sum()
                for p in tabn.beta_mlp.parameters():
                    reg = reg + p.pow(2).sum()
        return reg
