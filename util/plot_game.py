import argparse
import matplotlib.pyplot as plt

def plot_game(logfile, out_path):
    history_a = []
    history_b = []
    sign = 1
    with open(logfile, 'r') as f:
        log_text = f.readlines()
        for line in log_text:
            if line.startswith('Round'):
                print(line)
                tokens = list(map(lambda x: x.strip(','), line.strip().split(' ')))
                assert len(tokens) == 6
                history_a.append(int(tokens[4-sign][1:-1]))
                history_b.append(int(tokens[4+sign][1:-1]))
                sign *= -1
            elif line.startswith('Final'):
                tokens = list(map(lambda x: x.strip(','), line.strip().split(' ')))
                assert len(tokens) == 5
                history_a.append(int(tokens[3-sign][1:-1]))
                history_b.append(int(tokens[3+sign][1:-1]))
                sign *= -1
    plt.plot(history_a, label='A')
    plt.plot(history_b, label='B')
    plt.xlabel('Round')
    plt.legend()
    plt.gcf().savefig(out_path, format='pdf')
    plt.show()
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, help='Path to logfile')
    parser.add_argument('--output', type=str, help='Output path')
    args = parser.parse_args()
    plot_game(args.logfile, args.output)
