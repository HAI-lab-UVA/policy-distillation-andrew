import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
import pandas as pd
import numpy as np
import gymnasium as gym

class ACPolicyDistillation:
    """
    Class containing a selection of Actor-Critic policy distillation techniques for baseline experimentation.
    """
    def __init__(self, args):
        # TODO: Initialize the experiment, including: teacher and student models, 
        # optimization method, distance metric, type of distillation, hyperparameters, etc
        self.args = args

        # Setup vectorized env for teacher and student, with the student having a separate task
        if args.env_name == 'pusher':
            self.teacher_env = gym.make("Pusher-v4")
            self.student_env = gym.make("envs.register:NewGoal-Pusher-v4")
        else:
            assert NotImplementedError, f"The environment {args.env_name} is not supported"
        self.state_shape = self.teacher_env.observation_space.shape or self.teacher_env.observation_space.n
        self.action_shape = self.teacher_env.action_space.shape or self.teacher_env.action_space.n

        # TODO: Initialize both teacher and student with pre-defined networks for actor and critic
        self.teacher_policy = ts.policy.TRPOPolicy(actor=ts.utils.net.continuous.Actor(),
                                                   critic=ts.utils.net.continuous.Critic(),
                                                   optim=torch.optim.Adam(),
                                                   dist_fn=None,
                                                   action_space=self.action_shape,
                                                   observation_space=self.state_shape,
                                                   max_kl=0.01,
                                                   discount_factor=0.99,
                                                   optim_critic_iters=5,
                                                   )
        
        self.student_policy = ts.policy.TRPOPolicy(actor=ts.utils.net.continuous.Actor(),
                                                   critic=ts.utils.net.continuous.Critic(),
                                                   optim=torch.optim.Adam(),
                                                   dist_fn=None,
                                                   action_space=self.action_shape,
                                                   observation_space=self.state_shape,
                                                   max_kl=0.01,
                                                   discount_factor=0.99,
                                                   optim_critic_iters=5,
                                                   )

        # TODO: Setup collectors for teacher and student
        # TODO: Setup trainer for teacher
        # TODO: Setup TB logging for teacher and student
        # TODO: Define objective func/distance metric

    def RunVanilla(self):
        """
        Run an experiment with vanilla policy distillation.
        """
        # TODO: Train teacher as per docs until convergence
        # TODO: Figure out how to train student according to objective function 
        # (https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#train-a-policy-with-customized-codes)
        assert NotImplementedError
    
    def RunBootstrap(self):
        """
        Run a policy distillation experiment where the value function is bootstrapped from the Teacher.
        """
        assert NotImplementedError
    
    def RunCriticReward(self):
        """
        Run a policy distillation experiment where the critic is used as intrinsic reward for the Student.
        """
        assert NotImplementedError
