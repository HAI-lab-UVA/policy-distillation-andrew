from parseargs import parse_arguments
from distillation import ACPolicyDistillation
import numpy as np
import torch

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    distillation = ACPolicyDistillation(args)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)