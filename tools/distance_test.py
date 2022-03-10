import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='read log file')
    parser.add_argument('--distance', type=float, default=1., help='distance')
    parser.add_argument('--sigma', type=float, default=0.5, help='sigma')
    args = parser.parse_args()
    return args

def dis_to_score(distance, sigma):
    dis = torch.exp(-torch.tensor(distance, dtype=torch.float) ** 2 / (2.0 * sigma ** 2))
    return dis

if __name__ == '__main__':
    args = parse_args()
    score = dis_to_score(args.distance, args.sigma)
    print(score)