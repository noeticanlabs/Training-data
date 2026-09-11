"""UDLM-v0.1 developmental seed.

A deliberately small byte-level recurrent dynamical language model built from:
- homogeneous node primitives
- sparse stateful directed relations
- fast carrier state X
- relational edge state R
- temporal trace H
- slow learned condition Theta
- fixed (non-learned) byte excitation
- one shared local evolution law
- minimal 259-way readout

No attention, GRU/LSTM cell, token embedding table, or specialist F0/F1/F2 modules.

Revision: the carrier and relation recurrences use bounded leaky interpolation
rather than repeatedly applying tanh to the accumulated state. This preserves
state-space headroom and keeps new byte excitation consequential.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

VOCAB_SIZE = 259
BOS = 256
EOS = 257
PAD = 258


@dataclass
class UDLMConfig:
    n_nodes: int = 16
    d_state: int = 16
    d_rel: int = 8
    d_hist: int = 8
    d_theta: int = 8
    hidden: int = 32
    micro_steps: int = 2
    vocab_size: int = VOCAB_SIZE
    input_gain: float = 3.0
    x_update_rate: float = 0.35
    rel_update_rate: float = 0.20


@dataclass
class UDLMState:
    x: Tensor
    r: Tensor
    h: Tensor

    def detach(self) -> "UDLMState":
        return UDLMState(self.x.detach(), self.r.detach(), self.h.detach())


class UDLM(nn.Module):
    def __init__(self, cfg: UDLMConfig = UDLMConfig()):
        super().__init__()
        self.cfg = cfg
        src, dst = self._make_sparse_topology(cfg.n_nodes)
        self.register_buffer("edge_src", src, persistent=True)
        self.register_buffer("edge_dst", dst, persistent=True)
        self.n_edges = int(src.numel())
        self.register_buffer("input_pattern", self._fixed_injection(cfg.n_nodes, cfg.d_state), persistent=True)
        self.theta = nn.Parameter(torch.zeros(cfg.n_nodes, cfg.d_theta))
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

        rel_in = cfg.d_rel + 2 * cfg.d_state + 2 * cfg.d_hist
        self.rel_update = nn.Sequential(nn.Linear(rel_in, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, cfg.d_rel))
        msg_in = cfg.d_state + cfg.d_rel + 2 * cfg.d_theta
        self.message = nn.Sequential(nn.Linear(msg_in, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, cfg.d_state))
        node_in = cfg.d_state + cfg.d_state + cfg.d_state + cfg.d_hist + cfg.d_theta
        self.node_update = nn.Sequential(nn.Linear(node_in, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, cfg.d_state))
        self.hist_update = nn.Sequential(nn.Linear(cfg.d_hist + cfg.d_state, cfg.hidden // 2), nn.Tanh(), nn.Linear(cfg.hidden // 2, cfg.d_hist))
        self.hist_gate_logit = nn.Parameter(torch.tensor(2.0))
        readout_in = 2 * cfg.d_state + cfg.d_hist + cfg.d_rel
        self.readout = nn.Linear(readout_in, cfg.vocab_size)

    @staticmethod
    def _make_sparse_topology(n: int) -> Tuple[Tensor, Tensor]:
        src, dst = [], []
        skip = 5 if n % 5 else 3
        for i in range(n):
            src.extend([i, i])
            dst.extend([(i + 1) % n, (i + skip) % n])
        return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

    @staticmethod
    def _fixed_injection(n: int, d: int) -> Tensor:
        i = torch.arange(n, dtype=torch.float32).unsqueeze(1) + 1.0
        j = torch.arange(d, dtype=torch.float32).unsqueeze(0) + 1.0
        p = torch.sin(i * j * 0.61803398875) + torch.cos(i * j * 0.41421356237)
        return F.normalize(p, dim=-1)

    def fixed_byte_excitation(self, token: Tensor) -> Tensor:
        c = self.cfg
        t = token.float().unsqueeze(-1) + 1.0
        k = torch.arange(c.d_state, device=token.device, dtype=torch.float32) + 1.0
        harmonic = torch.sin(t * k * 0.1732050808) + torch.cos(t * k * 0.1131370849)
        bit_idx = (torch.arange(c.d_state, device=token.device) % 9).long()
        bits = (((token.unsqueeze(-1) >> bit_idx) & 1).float() * 2.0 - 1.0)
        return F.normalize(harmonic + 0.5 * bits, dim=-1)

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> UDLMState:
        c = self.cfg
        device = device or self.theta.device
        return UDLMState(
            x=torch.zeros(batch_size, c.n_nodes, c.d_state, device=device),
            r=torch.zeros(batch_size, self.n_edges, c.d_rel, device=device),
            h=torch.zeros(batch_size, c.n_nodes, c.d_hist, device=device),
        )

    def step(self, token: Tensor, state: UDLMState, *, reset_rel: bool = False, reset_hist: bool = False):
        c = self.cfg
        x, r, h = state.x, state.r, state.h
        if reset_rel:
            r = torch.zeros_like(r)
        if reset_hist:
            h = torch.zeros_like(h)

        exc = self.fixed_byte_excitation(token)
        external = c.input_gain * exc.unsqueeze(1) * self.input_pattern.unsqueeze(0)
        theta = self.theta.unsqueeze(0).expand(x.size(0), -1, -1)
        src, dst = self.edge_src, self.edge_dst
        alpha_x = float(c.x_update_rate)
        alpha_r = float(c.rel_update_rate)

        for _ in range(c.micro_steps):
            x_s, x_d = x[:, src], x[:, dst]
            h_s, h_d = h[:, src], h[:, dst]
            dr = self.rel_update(torch.cat([r, x_s, x_d, h_s, h_d], dim=-1))
            r_candidate = torch.tanh(dr)
            r = (1.0 - alpha_r) * r + alpha_r * r_candidate
            th_s, th_d = theta[:, src], theta[:, dst]
            msg_e = self.message(torch.cat([x_s, r, th_s, th_d], dim=-1))
            incoming = torch.zeros_like(x)
            idx = dst.view(1, -1, 1).expand(x.size(0), -1, c.d_state)
            incoming.scatter_add_(1, idx, msg_e)
            degree = torch.zeros(c.n_nodes, device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            incoming = incoming / degree.clamp_min(1.0).view(1, -1, 1)
            proposal_raw = self.node_update(torch.cat([x, incoming, external, h, theta], dim=-1))
            x_candidate = torch.tanh(proposal_raw)
            x = (1.0 - alpha_x) * x + alpha_x * x_candidate
            h_prop = self.hist_update(torch.cat([h, x], dim=-1))
            keep = torch.sigmoid(self.hist_gate_logit)
            h = keep * h + (1.0 - keep) * torch.tanh(h_prop)

        z = torch.cat([x.mean(dim=1), x.amax(dim=1), h.mean(dim=1), r.mean(dim=1)], dim=-1)
        logits = self.readout(z)
        return logits, UDLMState(x=x, r=r, h=h)

    def forward(self, tokens: Tensor, state: Optional[UDLMState] = None, *, reset_rel_each_token: bool = False, reset_hist_each_token: bool = False):
        if state is None:
            state = self.initial_state(tokens.size(0), tokens.device)
        outs = []
        for t in range(tokens.size(1)):
            logits, state = self.step(tokens[:, t], state, reset_rel=reset_rel_each_token, reset_hist=reset_hist_each_token)
            outs.append(logits)
        return torch.stack(outs, dim=1), state

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new_bytes: int = 128, *, greedy: bool = True, temperature: float = 1.0) -> bytes:
        self.eval()
        state = self.initial_state(1, self.theta.device)
        logits = None
        for tok in [BOS] + list(prompt):
            token = torch.tensor([tok], device=self.theta.device, dtype=torch.long)
            logits, state = self.step(token, state)
        out = bytearray()
        for _ in range(max_new_bytes):
            assert logits is not None
            if greedy:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            if nxt == EOS:
                break
            if 0 <= nxt <= 255:
                out.append(nxt)
            token = torch.tensor([nxt], device=self.theta.device, dtype=torch.long)
            logits, state = self.step(token, state)
        return bytes(out)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
