import argparse

import numpy as np
from matplotlib import pyplot as plt

# plt.style.use('seaborn-whitegrid')
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn-deep')

# large = 18
# med = 12
# small = 10
# params = {
#     'legend.fontsize': small,
#     'figure.figsize': (16, 10),
#     'axes.labelsize': med,
#     'axes.titlesize': med,
#     'xtick.labelsize': small,
#     'ytick.labelsize': small,
#     'figure.titlesize': large,
# }
# plt.rcParams.update(params)


def read_file(filename):
    header = []
    content = []
    with open(filename, 'r', encoding='utf8') as f:
        is_first_line = True
        for line in f:
            line = line.strip()
            if line[-1] == ',':
                line = line[:-1]
            line = line.split(',')
            if is_first_line:
                header = line
            else:
                numbers = list(map(float, line))
                content.append(numbers)
            is_first_line = False
    return header, content


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time performance")
    parser.add_argument("--filename", type=str, default='times.txt')
    args = parser.parse_args()

    methods, times = read_file(args.filename)

    fig, ax = plt.subplots()
    # ax.grid(b=True, which='major')
    ax.set_axisbelow(True)

    # Show the major grid lines with dark grey lines
    # ax.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)

    # Show the minor grid lines with very faint and almost transparent lines
    # ax.minorticks_on()
    # ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # list of colors
    all_colors = list(plt.cm.colors.cnames.keys())

    times = np.array(times)
    # x = np.array([2 ** i for i in range(3, 3+len(times))])
    x = np.array([50*i for i in range(1, 50)])
    # x = np.log(x)
    for i in range(times.shape[1]):
        name = methods[i]
        # color = all_colors[5 + i//2]
        # if name.endswith('lstm'):
        #     continue
        y = times[:, i]
        ax.plot(x[5:], y[5:], label=name, linestyle='dashed' if i > 1 else 'solid')

    ax.set_title('Attn performance')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Time (s)')
    # plt.autoscale(axis='y')
    # ax.set_ylim(0.0, np.max(times)+1)
    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    ax.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig(args.output_path)
    plt.show()
