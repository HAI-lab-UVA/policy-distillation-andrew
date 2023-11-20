class ACPolicyDistillation:
    """
    Class containing a selection of Actor-Critic policy distillation techniques for baseline experimentation.
    """
    def __init__(self, args):
        # TODO: Initialize the experiment, including: teacher and student models, 
        # optimization method, distance metric, type of distillation, hyperparameters, etc
        self.args = args
        # TODO: Setup vectorized env for teacher and student, with the student having a separate task
        # TODO: Initialize both teacher and student with pre-defined networks for actor and critic
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
