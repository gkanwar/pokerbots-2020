import argparse
import os
import sys

sys.path.append(os.getcwd())
import config as cfg
import matplotlib.pyplot as plt
import engine

topdir = os.getcwd()

def run_match(ind, bot1, bot2, count):
    cfg.PLAYER_1_PATH = topdir + '/robots/' + bot1
    cfg.PLAYER_2_PATH = topdir + '/robots/' + bot2
    os.mkdir(f'match{ind}')
    os.chdir(f'match{ind}')
    for k in range(count):
        engine.Game().run()
        os.rename('A.txt', f'{k}_A.txt')
        os.rename('B.txt', f'{k}_B.txt')
        os.rename('gamelog.txt', f'{k}_gamelog.txt')
        os.rename('gamelog.pdf', f'{k}_gamelog.pdf')
        plt.gca().set_prop_cycle(None)
    os.chdir('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Tournament outfile directory')
    parser.add_argument('--count', type=int, help='How many times to run matchup')
    parser.add_argument('bot', nargs='+', help='Bots to enter the tournament')
    args = parser.parse_args()
    bots = args.bot
    os.makedirs(args.path, exist_ok=True)
    os.chdir(args.path)
    ind = 1
    for i in range(len(bots)):
        for j in range(i+1, len(bots)):
            print('Match {:d}: {} vs {}'.format(ind, bots[i], bots[j]))
            run_match(ind, bots[i], bots[j], args.count)
            ind += 1
    print(f'Tournament complete, outputs are in {args.path}')
