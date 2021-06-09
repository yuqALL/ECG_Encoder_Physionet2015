from datasets.data_read_utils import load_record
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.patches as mpatches
from sys import exit
import copy

plt.rc('font', family='Times New Roman')


def gaussion_noise(sig):
    cnt = copy.deepcopy(sig)
    center = np.mean(cnt)
    print(center)
    cnt -= center
    return cnt


def minmax_scale(cnt):
    # minv = np.nanmin(cnt, axis=0)
    # maxv = np.nanmax(cnt, axis=0)
    minv = np.percentile(cnt, 5, axis=0)
    maxv = np.percentile(cnt, 95, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    cnt = (cnt - 0.5) * 2
    return cnt, minv, maxv


if __name__ == '__main__':
    sig1 = load_record('./a103l')
    sig2 = load_record('./a109l')
    # 取5s数据演示添加噪声
    ecg = sig2.p_signal[:, 0].T[73750:75000]

    y_ticks = 'Amplitude(mV)'

    t = [i for i in range(1250)]

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.align_labels()

    axs[0].plot(t, ecg, color='black')

    axs[0].set_xlim(0, 1250)
    # axs[0].set_xlabel('time')
    axs[0].set_title('Raw ECG', fontsize=12)
    axs[0].set_ylabel(y_ticks)
    labels = ['4:55', '4:51', '4:52', '4:53', '4:54', '5:00']
    axs[0].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[0].set_xticklabels(labels)
    # axs[0].legend()

    # cnt = gaussion_noise(ecg)
    cnt, _, _ = minmax_scale(ecg)
    axs[1].plot(t, cnt, color='black')
    axs[1].set_xlim(0, 1250)
    # axs[1].set_xlabel('time')
    axs[1].set_title('norm', fontsize=12)
    axs[1].set_ylabel(y_ticks)
    axs[1].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[1].set_xticklabels(labels)

    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.3)
    plt.savefig("norm_image.png", dpi=200, bbox_inches='tight')
    plt.show()
    pass
