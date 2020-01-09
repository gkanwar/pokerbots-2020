import argparse
import os
import sys

sys.path.append(os.getcwd())
import config as cfg
import engine

topdir = os.getcwd()

def run_match(ind, bot1, bot2):
    cfg.PLAYER_1_PATH = topdir + '/' + bot1
    cfg.PLAYER_2_PATH = topdir + '/' + bot2
    os.mkdir(f'match{ind}')
    os.chdir(f'match{ind}')
    engine.Game().run()
    os.chdir('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Tournament outfile directory')
    parser.add_argument('bot', nargs='+', help='Bots to enter the tournament')
    args = parser.parse_args()
    bots = args.bot
    os.makedirs(args.path, exist_ok=True)
    os.chdir(args.path)
    ind = 1
    for i in range(len(bots)):
        for j in range(i+1, len(bots)):
            print('Match {:d}: {} vs {}'.format(ind, bots[i], bots[j]))
            run_match(ind, bots[i], bots[j])
            ind += 1
    print(f'Tournament complete, outputs are in {args.path}')
