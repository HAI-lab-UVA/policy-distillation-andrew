import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

import tianshou as ts
from tianshou.data import Batch
from tianshou.policy.base import MultipleLRSchedulers

TDistributionFunction: TypeAlias = Callable[..., torch.distributions.Distribution]
TLearningRateScheduler: TypeAlias = torch.optim.lr_scheduler.LRScheduler | MultipleLRSchedulers

class VanillaStudentPolicy(ts.policy.A2CPolicy):
    """
    Subclass of tianshou.policy.A2CPolicy to override network update functions in learn().
    Code adapted from https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/trpo.py
    """
    
    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: TDistributionFunction,
        action_space: gym.Space,
        teacher_policy: ts.policy.TRPOPolicy,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            vf_coef=None,  # type: ignore
            ent_coef=None,  # type: ignore
            max_grad_norm=None,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.teacher_policy = teacher_policy

    def learn(  
        self,
        batch: Batch,
        batch_size: int | None,
        repeat: int,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Update parameters with vanilla on-policy distillation.

        Loss: H(pi(s)||pi_theta(s))*[V_pi(s)-V_pi_theta(s)]_{>0}
        """

        val_diffs, kls, dist_losses = [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # Get pi(s)
                # TODO: Check whether state should be None
                with torch.no_grad():
                    teacher_dist = self.teacher_policy(minibatch).dist

                # Get pi_theta(s)
                # TODO: Check whether state should be None
                student_dist = self(minibatch).dist

                # Calculate H(pi(s)||pi_theta(s)) where H is KL-Divergence between two distributions over actions
                h = kl_divergence(teacher_dist, student_dist)
                kls.append(h.mean().item())

                # Get V_pi(s)
                with torch.no_grad():
                    t_val = self.teacher_policy.critic(minibatch.obs).flatten()

                # Get V_pi_theta(s)
                s_val = self.critic(minibatch.obs).flatten()

                # Take difference of values
                val_diff = t_val - s_val
                # If dif > 0, set to 1 as per https://arxiv.org/pdf/1902.02186.pdf pg 7
                val_diff.where(torch.gt(val_diff, 0.0), torch.tensor(1.0))
                val_diffs.append(val_diff.mean().item())

                dist_loss = h * val_diff
                avg_dist_loss = dist_loss.mean()
                dist_losses.append(avg_dist_loss.item())

                self.optim.zero_grad()
                avg_dist_loss.backward()
                self.optim.step()

        return {
            "val_diff": val_diffs,
            "kl": kls,
            "loss/distill": dist_losses,
        }
