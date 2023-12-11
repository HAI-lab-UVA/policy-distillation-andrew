import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import TRPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

import pandas as pd
import numpy as np
import os
import gymnasium as gym

class ACPolicyDistillation:
    """
    Class containing a selection of Actor-Critic policy distillation techniques for baseline experimentation. 
    Based on code from https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/mujoco_trpo.py
    """
    def dist(self, *logits):
        """Defines the distribution function for computing the action"""
        return Independent(Normal(*logits), 1)
    
    def make_AC(self):
        """
        Initializes Actor and Critic networks.
        """
        net_a = Net(
            self.state_shape,
            hidden_sizes=self.args.hidden_sizes,
            activation=nn.Tanh,
            device=self.device,
        )
        actor = ActorProb(
            net_a,
            self.action_shape,
            unbounded=True,
            device=self.device,
        ).to(self.device)

        net_c = Net(
            self.state_shape,
            hidden_sizes=self.args.hidden_sizes,
            activation=nn.Tanh,
            device=self.device,
        )
        critic = Critic(net_c, device=self.device).to(self.device)
        torch.nn.init.constant_(actor.sigma_param, -0.5)
        for m in list(actor.modules()) + list(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        optim = torch.optim.Adam(critic.parameters(), lr=self.args.lr)
        lr_scheduler = None
        if self.args.lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(self.args.step_per_epoch / self.args.step_per_collect) * self.args.epoch

            lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
        
        return {'actor': actor, 
                'critic': critic, 
                'optim': optim, 
                'schedule': lr_scheduler}

    def __init__(self, args):
        # TODO: Initialize the experiment, including: teacher and student models, 
        # optimization method, distance metric, type of distillation, hyperparameters, etc
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup vectorized env for teacher and student, with the student having a separate task
        # TODO: Use ts DummyVecEnv
        if args.env_name == 'pusher':
            self.env = gym.make("Pusher-v4")
            self.teacher_train_env = ts.env.ShmemVectorEnv([lambda: gym.make("Pusher-v4") for _ in range(args.training_num)])            
            self.teacher_test_env = ts.env.ShmemVectorEnv([lambda: gym.make("Pusher-v4") for _ in range(self.args.test_num)])            
            self.student_train_env = ts.env.ShmemVectorEnv([lambda: gym.make("envs.register:NewGoal-Pusher-v4") for _ in range(args.training_num)])
            self.student_test_env = ts.env.ShmemVectorEnv([lambda: gym.make("envs.register:NewGoal-Pusher-v4") for _ in range(args.test_num)])
        else:
            assert NotImplementedError, f"The environment {args.env_name} is not supported"

        # set env seeds
        self.teacher_train_env.seed(self.args.seed)
        self.teacher_test_env.seed(self.args.seed)
        self.student_train_env.seed(self.args.seed)
        self.student_test_env.seed(self.args.seed)

        # Norm env observations
        self.teacher_train_env = ts.env.VectorEnvNormObs(self.teacher_train_env)
        self.teacher_test_env = ts.env.VectorEnvNormObs(self.teacher_test_env, update_obs_rms=False)
        self.teacher_test_env.set_obs_rms(self.teacher_train_env.get_obs_rms())
        self.student_train_env = ts.env.VectorEnvNormObs(self.student_train_env)
        self.student_test_env = ts.env.VectorEnvNormObs(self.student_test_env, update_obs_rms=False)
        self.student_test_env.set_obs_rms(self.student_train_env.get_obs_rms())
        
        self.state_shape = self.env.observation_space.shape or self.env.observation_space.n
        self.action_shape = self.env.action_space.shape or self.env.action_space.n

        # Initialize both teacher and student with pre-defined networks for actor and critic
        self.teacher_ac = self.make_AC()
        self.student_ac = self.make_AC()

        self.teacher_policy = TRPOPolicy(
            actor=self.teacher_ac['actor'],
            critic=self.teacher_ac['critic'],
            optim=self.teacher_ac['optim'],
            dist_fn=self.dist,
            discount_factor=args.gamma,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            action_scaling=True,
            action_bound_method=args.bound_action_method,
            lr_scheduler=self.teacher_ac['schedule'],
            action_space=self.env.action_space,
            advantage_normalization=args.norm_adv,
            optim_critic_iters=args.optim_critic_iters,
            max_kl=args.max_kl,
            backtrack_coeff=args.backtrack_coeff,
            max_backtracks=args.max_backtracks,
        )

        self.student_policy = TRPOPolicy(
            actor=self.student_ac['actor'],
            critic=self.student_ac['critic'],
            optim=self.student_ac['optim'],
            dist_fn=self.dist,
            discount_factor=args.gamma,
            gae_lambda=args.gae_lambda,
            reward_normalization=args.rew_norm,
            action_scaling=True,
            action_bound_method=args.bound_action_method,
            lr_scheduler=self.student_ac['optim'],
            action_space=self.env.action_space,
            advantage_normalization=args.norm_adv,
            optim_critic_iters=args.optim_critic_iters,
            max_kl=args.max_kl,
            backtrack_coeff=args.backtrack_coeff,
            max_backtracks=args.max_backtracks,
        )

        # Setup buffers and collectors for teacher and student
        if args.training_num > 1:
            self.teacher_buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
            self.student_buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
        else:
            self.teacher_buffer = ReplayBuffer(args.buffer_size)
            self.student_buffer = ReplayBuffer(args.buffer_size)

        self.teacher_train_collector = Collector(self.teacher_policy, self.teacher_train_env, self.teacher_buffer, exploration_noise=True)
        self.teacher_test_collector = Collector(self.teacher_policy, self.teacher_test_env)
        self.student_train_collector = Collector(self.student_policy, self.student_train_env, self.student_buffer, exploration_noise=True)
        self.student_test_collector = Collector(self.student_policy, self.student_test_env)

        # Setup TB logging for teacher and student
        teacher_path = os.path.join('./logging/teacher', args.save_path)
        self.teacher_writer = SummaryWriter(teacher_path)
        self.teacher_writer.add_text("args", str(self.args))
        self.teacher_logger = TensorboardLogger(self.teacher_writer)

        student_path = os.path.join('./logging/student', args.save_path)
        self.student_writer = SummaryWriter(student_path)
        self.student_writer.add_text("args", str(self.args))
        self.student_logger = TensorboardLogger(self.student_writer)

        # Define objective func/distance metric

        # Setup trainer for teacher
        self.teacher_trainer = OnpolicyTrainer(
            policy=self.teacher_policy,
            train_collector=self.teacher_train_collector,
            test_collector=self.teacher_test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=1,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            logger=self.teacher_logger,
            test_in_train=False,
        )

        if args.retrain_teacher: 
            # Train teacher as per docs until convergence
            self.teacher_results = self.teacher_trainer.run()
            torch.save(self.teacher_policy.state_dict(), './saved_models/trpo/policy.pt')
        else: 
            self.teacher_policy.load_state_dict(torch.load('./saved_models/trpo/policy.pt'))
        # TODO: Run the appropriate experiment
        if args.distill_method == 'vanilla':
            self.RunVanilla()
        else:
            assert NotImplementedError, f"The distillation method {args.distil_method} is not supported"
        return None

    def RunVanilla(self):
        """
        Run an experiment with vanilla on-policy distillation.

        Loss: H(pi(s)||pi_theta(s))*[V_pi(s)-V_pi_theta(s)]_{>0}
        """
        # TODO: Figure out how to train student according to objective function 
        # (https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#train-a-policy-with-customized-codes)
        # * Collect experience normally
        # * Update AC NNs w/ loss directly, instead of student TRPOPolicy learn() or update()
        # * Test as normal

        # TODO: See what would happen if we kept this loop the same but instead of calling update we 
        # run the optimizer with respect to the above loss

        # pre-collect at least 5000 transitions with random action before training
        self.student_train_collector.collect(n_step=5000, random=True)

        # self.student_policy.set_eps(0.1)
        for i in range(int(1e6)):  # total step
            collect_result = self.student_train_collector.collect(n_step=10)

            # once if the collected episodes' mean returns reach the threshold,
            # or every 1000 steps, we test it on test_collector
            if collect_result['rews'].mean() >= i % 1000 == 0:
                # self.student_policy.set_eps(0.05)
                result = self.student_test_collector.collect(n_episode=100)
                self.teacher_writer.add_scalar("Reward/test", result['rews'].mean(), i)
                if result['rews'].mean() >= self.student_env.spec.reward_threshold:
                    print(f'Finished training! Test mean returns: {result["rews"].mean()}')
                    break
                else:
                    # back to training eps
                    self.student_policy.set_eps(0.1)

            # train policy with a sampled batch data from buffer
            batch, indices = self.student_buffer.sample(self.args.batch_size)
            batch = self.student_policy.process_fn(batch, self.student_buffer, indices)
            dist_losses = []
            split_batch_size = self.args.batch_size or -1
            for _ in range(self.args.repeat_per_collect):
                for minibatch in batch.split(split_batch_size, merge_last=True):
                    # Get pi(s)
                    # TODO: Check whether state should be None
                    with torch.no_grad():
                        t_logits, t_hidden = self.teacher_policy.actor(minibatch.obs, state=None, info=minibatch.info)
                        if isinstance(t_logits, tuple):
                            t_dist = self.teacher_policy.dist_fn(*t_logits)
                        else:
                            t_dist = self.teacher_policy.dist_fn(t_logits)

                    # Get pi_theta(s)
                    # TODO: Check whether state should be None
                    s_logits, s_hidden = self.student_policy.actor(minibatch.obs, state=None, info=minibatch.info)
                    if isinstance(s_logits, tuple):
                        s_dist = self.student_policy.dist_fn(*s_logits)
                    else:
                        s_dist = self.student_policy.dist_fn(s_logits)

                    # Calculate H(pi(s)||pi_theta(s)) where H is Shannon’s cross entropy between two distributions over actions
                    # BUG: TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not Independent
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
                    self.student_policy.optim.step()
            self.teacher_writer.add_scalar("Loss/train", np.mean(dist_losses), i)
    
    def RunBootstrap(self):
        """
        Run a policy distillation experiment where the value function is bootstrapped from the Teacher.

        Loss: -log(pi_theta(a_t|T_t)) * [r(a_t|T_t) + V_pi(T_{t+1})]
        """
        assert NotImplementedError
    
    def RunCriticReward(self):
        """
        Run a policy distillation experiment where the critic is used as intrinsic reward for the Student.

        Loss: E_pi_theta[SUM over T: V_pi(T_{t+1}) - V_pi(T_t) + r_t]
        """
        assert NotImplementedError
