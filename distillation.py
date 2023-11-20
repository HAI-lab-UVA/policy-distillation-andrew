class ACPolicyDistillation:
    """
    Class containing a selection of Actor-Critic policy distillation techniques for baseline experimentation.
    """
    def __init__(self, args):
        # TODO: Initialize the experiment, including: teacher and student models, 
        # optimization method, distance metric, type of distillation, hyperparameters, etc
        self.args = args

    def RunVanilla(self):
        """
        Run an experiment with vanilla policy distillation.
        """        
        return None
    
    def RunBootstrap(self):
        """
        Run a policy distillation experiment where the value function is bootstrapped from the Teacher.
        """
        return None
    
    def RunCriticReward(self):
        """
        Run a policy distillation experiment where the critic is used as intrinsic reward for the Student.
        """
        return None
