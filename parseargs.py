import argparse
import os 
import datetime
import distutils

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--save-path", type=str, default=f'{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}',
                        help="the directory to which results are saved")
    parser.add_argument("--env-name", type=str, default="pusher",
                        help="the name of the environment")
    parser.add_argument("--distill-method", type=str, default="vanilla",
                        help="the method of distillation to use")
    parser.add_argument("--retrain-teacher", type=lambda x:bool(distutils.util.strtobool(x)), default=False,
                        help="whether to retrain and save teacher policy")
    
    # Below args from https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/mujoco_trpo.py
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="*",
        default=[64, 64],
    )  # baselines [32, 32]
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=1024)
    parser.add_argument("--repeat-per-collect", type=int, default=1)
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=10)
    # batch-size >> step-per-collect means calculating all data in one singe forward.
    parser.add_argument("--batch-size", type=int, default=99999)
    # trpo special
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    # TODO tanh support
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--optim-critic-iters", type=int, default=20)
    parser.add_argument("--max-kl", type=float, default=0.01)
    parser.add_argument("--backtrack-coeff", type=float, default=0.8)
    parser.add_argument("--max-backtracks", type=int, default=10)
    
    args = parser.parse_args()

    return args