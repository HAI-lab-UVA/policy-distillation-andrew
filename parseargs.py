import argparse
import os 
import time

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--save-path", type=str, default=f'./logging/{time.time()}',
                        help="the directory to which results are saved")
    
    args = parser.parse_args()

    return args