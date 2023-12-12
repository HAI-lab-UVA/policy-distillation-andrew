from collections.abc import Callable
from typing import Any, Literal, Optional, Tuple, TypeAlias, Union
import numpy as np

import gymnasium as gym
import torch
from torch.distributions import kl_divergence

import tianshou as ts
from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy.base import MultipleLRSchedulers, BasePolicy, _gae_return

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

        val_diffs, hs, dist_losses = [], [], []
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

                # Calculate H(pi(s)||pi_theta(s)) where H is the KL-Divergence between two distributions over actions
                h = kl_divergence(teacher_dist, student_dist)
                hs.append(h.mean().item())

                # Get V_pi(s)
                with torch.no_grad():
                    t_val = self.teacher_policy.critic(minibatch.obs).flatten()

                # Get V_pi_theta(s)
                s_val = self.critic(minibatch.obs).flatten()

                # Take difference of values
                val_diff = t_val - s_val
                # If dif > 0, set to 1 as per https://arxiv.org/pdf/1902.02186.pdf pg 7
                val_diff = torch.where(torch.gt(val_diff, 0.0), torch.tensor(1.0), val_diff)
                val_diffs.append(val_diff.mean().item())

                dist_loss = h * val_diff
                avg_dist_loss = dist_loss.mean()
                dist_losses.append(avg_dist_loss.item())

                self.optim.zero_grad()
                avg_dist_loss.backward()
                self.optim.step()

        return {
            "val_diff": val_diffs,
            "h": hs,
            "loss/distill": dist_losses,
        }

class TeacherVStudentPolicy(ts.policy.TRPOPolicy):
    """
    Subclass of tianshou.policy.TRPOPolicy to override return calculations in compute_episodic_returns() and use custom reward function.
    Code adapted from https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/base.py
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
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        discount_factor: float = 0.99,
        # TODO: rename to return_normalization?
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
            max_kl=max_kl,
            max_backtracks=max_backtracks,
            backtrack_coeff=backtrack_coeff,
            optim_critic_iters=optim_critic_iters,
            actor_step_size=actor_step_size,
            advantage_normalization=advantage_normalization,
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

    def compute_episodic_return(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch.

        :param Batch batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param numpy.ndarray indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        :param np.ndarray v_s_: the value function of all next states :math:`V(s')`.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage Estimation,
            should be in [0, 1]. Default to 0.95.

        :return: two numpy arrays (returns, advantage) with each shape (bsz, ).
        """
        with torch.no_grad():
            t_val = self.teacher_policy.critic(batch.obs).flatten()
            mask = np.repeat(self.value_mask(buffer, indices)[...,None],batch.obs_next.shape[1],axis=1)
            t_next_val = self.teacher_policy.critic(batch.obs_next*mask).flatten()

        # TODO: KL div and actor loss start and remain low throughout training
        # * behavior is similar to baseline 
        # * Check whether rew needs to be modified when saved to buffer
        rew = to_numpy(t_next_val) - to_numpy(t_val) + batch.rew
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * self.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage