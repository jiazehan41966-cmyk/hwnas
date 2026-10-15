"""Reuse the repository MLP and reward; adapt only the canonical decision ABI.

Channel/depth/kernel head names are legacy tensor names, reinterpreted here as
architecture/profile/mapping logits. This is a new strategy, not old V2 results.
The product distribution is conditioned on the exact available joint mask; its
normalized log probability (not unmasked marginal logs) drives REINFORCE.
"""
from __future__ import annotations

import math
import torch
from hwnas_fpga.search.rl_searcher import ActionSpace, Controller, RewardFunction
from tools.sonar_joint_v2 import policy


class JointController:
    def __init__(self, cfg, architecture_count=72):
        self.cfg = cfg
        self.shape = (architecture_count, len(policy.PROFILES), len(policy.POLICIES))
        # A private fork prevents changing a caller's global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"])
            self.model = Controller(ActionSpace(tuple(range(architecture_count)),
                tuple(range(3)), tuple(range(6)), (1,), ("fixed",)), max_stages=1,
                max_blocks_per_stage=1)
        self.generator = torch.Generator().manual_seed(cfg["seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.baseline, self.updates = 0., 0
        limits = cfg["hardware_budget"]["limits"]
        self.reward = RewardFunction(**cfg["reward"], normalization_scales=dict(
            accuracy=1., latency=limits["Latency"] / 1000., lut=limits["LUT"],
            dsp=limits["DSP"], bram=limits["BRAM"], energy=1.))

    def distribution(self, available):
        if len(available) != math.prod(self.shape) or not any(available):
            raise ValueError("Nonempty canonical joint mask required")
        heads = self.model(0, 0)
        logits = (heads["channel"][:, None, None] + heads["depth"][None, :, None]
                  + heads["kernel"][None, None, :]).reshape(-1)
        logits = logits.masked_fill(~torch.tensor(available, dtype=torch.bool), -torch.inf)
        return torch.distributions.Categorical(logits=logits)

    def sample(self, available):
        distribution = self.distribution(available)
        index = int(torch.multinomial(distribution.probs.detach(), 1, generator=self.generator))
        return index, distribution.log_prob(torch.tensor(index)), distribution.entropy()

    def compute_reward(self, cost, accuracy=None, *, feasible):
        if any(not policy.finite(cost.get(k)) for k in policy.METRICS):
            raise ValueError("Missing cost is not an infeasibility penalty")
        if feasible:
            policy.score(accuracy)
        # energy_mj=0 is an ignored ABI placeholder, never exported as measured energy.
        return self.reward.compute_reward(accuracy if feasible else 0., cost["Latency"] / 1000.,
            0., cost["DSP"], cost["BRAM"], cost["LUT"], is_feasible=feasible)

    def update(self, log_probability, entropy, reward):
        if not math.isfinite(reward):
            raise ValueError("Finite reward required")
        self.optimizer.zero_grad()
        advantage = reward - self.baseline
        loss = -advantage * log_probability - self.cfg["entropy_weight"] * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        self.optimizer.step()
        self.baseline = self.cfg["baseline_momentum"] * self.baseline + (1 - self.cfg["baseline_momentum"]) * reward
        self.updates += 1
        return float(loss.detach())

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    generator=self.generator.get_state(), baseline=self.baseline, updates=self.updates)

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.baseline, self.updates = state["baseline"], state["updates"]
