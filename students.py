import warnings
from typing import Any, Literal

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

import tianshou as ts
from tianshou.data import Batch
from tianshou.policy.base import TLearningRateScheduler
from tianshou.policy.modelfree.pg import TDistributionFunction

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

        actor_losses, vf_losses, step_sizes, kls, dist_losses = [], [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # Get pi(s)
                # TODO: Check whether state should be None
                # TODO: ActorProb outputs a touple of ((mu, sigma), state) 
                # where mu and sigma are the parameters of a gaussian dist.
                # This is why we need to call dist_fn: to make a distribution from these params.
                # Can we call log_prob on the torch.distributions?
                # TODO: Follow steps to compute KL div in trpo instead
                with torch.no_grad():
                    t_logits, t_hidden = self.teacher_policy.actor(minibatch.obs, state=None, info=minibatch.info)
                    if isinstance(t_logits, tuple):
                        t_dist = self.teacher_policy.dist_fn(*t_logits)
                    else:
                        t_dist = self.teacher_policy.dist_fn(t_logits)

                # Get pi_theta(s)
                # TODO: Check whether state should be None
                s_logits, s_hidden, _ = self.student_policy.actor(minibatch.obs, state=None, info=minibatch.info)
                if isinstance(s_logits, tuple):
                    s_dist = self.student_policy.dist_fn(*s_logits)
                else:
                    s_dist = self.student_policy.dist_fn(s_logits)

                # Calculate H(pi(s)||pi_theta(s)) where H is Shannon’s cross entropy between two distributions over actions
                # BUG: TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not Independent
                # TODO: This should probably be KL divergence
                h = torch.nn.functional.cross_entropy(s_dist, t_dist)

                # Get V_pi(s)
                with torch.no_grad():
                    t_val = self.teacher_policy.critic(minibatch.obs).flatten()

                # Get V_pi_theta(s)
                s_val = self.student_policy.critic(minibatch.obs).flatten()

                # Take difference of values
                val_diff = t_val - s_val
                # If dif > 0, set to 1 as per https://arxiv.org/pdf/1902.02186.pdf pg 7
                val_diff.where(torch.gt(val_diff, 0.0), torch.tensor(1.0))

                dist_loss = h * val_diff
                dist_losses.append(dist_loss)

                self.student_policy.optim.zero_grad()
                dist_loss.backward()
                # TODO: Make optimizer update both critic and actor
                self.student_policy.optim.step()

        return {
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "step_size": step_sizes,
            "kl": kls,
            "loss/distill": dist_losses,
        }
