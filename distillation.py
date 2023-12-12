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
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic

import pandas as pd
import numpy as np
import os
import gymnasium as gym

from students import VanillaStudentPolicy, TeacherVStudentPolicy

class ACPolicyDistillation:
    """
    Class containing a selection of Actor-Critic policy distillation techniques for baseline experimentation. 
    Based on code from https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/mujoco_trpo.py
    """
    def dist(self, *logits):
        """Defines the distribution function for computing the action"""
        return Independent(Normal(*logits), 1)
    
    def make_AC(self, comb_params=False):
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

        if comb_params:
            actor_critic = ActorCritic(actor, critic)
            for m in actor_critic.modules():
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

            optim = torch.optim.Adam(actor_critic.parameters(), lr=self.args.lr)
        else:
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
        self.teacher_ac = self.make_AC(comb_params=False)
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

        if args.distill_method == 'vanilla':
            self.student_ac = self.make_AC(comb_params=True)
            self.student_policy = VanillaStudentPolicy(
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
                teacher_policy = self.teacher_policy
            )
        elif args.distill_method == 'teacherV':
            self.student_ac = self.make_AC(comb_params=False)
            self.student_policy = TeacherVStudentPolicy(
                actor=self.student_ac['actor'],
                critic=self.student_ac['critic'],
                optim=self.student_ac['optim'],
                dist_fn=self.dist,
                discount_factor=args.gamma,
                gae_lambda=args.gae_lambda,
                reward_normalization=args.rew_norm,
                action_scaling=True,
                action_bound_method=args.bound_action_method,
                lr_scheduler=self.student_ac['schedule'],
                action_space=self.env.action_space,
                advantage_normalization=args.norm_adv,
                optim_critic_iters=args.optim_critic_iters,
                max_kl=args.max_kl,
                backtrack_coeff=args.backtrack_coeff,
                max_backtracks=args.max_backtracks,
                teacher_policy = self.teacher_policy
            )
        elif args.distill_method == 'baseline':
            self.student_ac = self.make_AC(comb_params=False)
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
                lr_scheduler=self.student_ac['schedule'],
                action_space=self.env.action_space,
                advantage_normalization=args.norm_adv,
                optim_critic_iters=args.optim_critic_iters,
                max_kl=args.max_kl,
                backtrack_coeff=args.backtrack_coeff,
                max_backtracks=args.max_backtracks,
            )
        else:
            assert NotImplementedError, f"The distillation method {args.distil_method} is not supported"

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

        # Train student 
        self.student_policy.actor.load_state_dict(self.teacher_policy.actor.state_dict())
        self.student_policy.critic.load_state_dict(self.teacher_policy.critic.state_dict())
        self.student_trainer = OnpolicyTrainer(
            policy=self.student_policy,
            train_collector=self.student_train_collector,
            test_collector=self.student_test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=1,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            logger=self.student_logger,
            test_in_train=False,
        )

        self.student_results = self.student_trainer.run()

        return None
    
    def RunBootstrap(self):
        """
        Run a policy distillation experiment where the value function is bootstrapped from the Teacher.

        Loss: -log(pi_theta(a_t|T_t)) * [r(a_t|T_t) + V_pi(T_{t+1})]
        """
        assert NotImplementedError
