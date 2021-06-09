from datasets.data_read_utils import load_record
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.patches as mpatches
from sys import exit
import copy

plt.rc('font', family='Times New Roman')


def gaussion_noise(sig, sigma='default', drop_noise=0.8):
    cnt = copy.deepcopy(sig)
    zeros = int(len(cnt) * drop_noise)
    ones = len(cnt) - zeros
    factor = np.array([0] * zeros + [1] * ones)
    np.random.shuffle(factor)
    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    noise = factor * noise
    cnt += noise
    return cnt, noise


def sin_gaussion_noise(sig, fz=50, factor='default', sigma='default'):
    cnt = copy.deepcopy(sig)
    continer = [i for i in range(cnt.shape[0])]
    if factor == 'default':
        factor = 0.05 * (np.nanmax(cnt) - np.nanmin(cnt))
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2 / 250 + np.random.randn(*cnt.shape) * np.pi)
    else:
        noise = factor * np.sin(np.array(continer) * fz * np.pi * 2 / 250)
    if sigma == 'default':
        sigma = 0.05 * (np.nanmax(cnt) - np.nanmin(cnt))
        noise2 = sigma * np.random.randn(*cnt.shape)
    else:
        noise2 = sigma * np.random.randn(*cnt.shape)
    cnt += noise
    cnt += noise2
    return cnt


if __name__ == '__main__':
    sig1 = load_record('./a103l')
    sig2 = load_record('./a109l')
    # 取5s数据演示添加噪声
    ecg = sig1.p_signal[:, 0].T[73750:75000]

    y_ticks = 'Amplitude(mV)'

    t = [i for i in range(1250)]

    fig, axs = plt.subplots(3, 1, sharex=True)
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

    cnt, noise = gaussion_noise(ecg)
    axs[1].plot(t, noise, color='black')
    axs[1].set_xlim(0, 1250)
    # axs[1].set_xlabel('time')
    axs[1].set_title('Noise', fontsize=12)
    axs[1].set_ylabel(y_ticks)
    axs[1].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[1].set_xticklabels(labels)
    # axs[1].legend()

    axs[2].plot(t, cnt, color='black')
    axs[2].set_xlim(0, 1250)
    axs[2].set_xlabel('time(mm:ss)', fontsize=14)
    axs[2].set_title('ECG with noise addition', fontsize=12)
    axs[2].set_ylabel(y_ticks)
    axs[2].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[2].set_xticklabels(labels)
    # axs[2].legend()

    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.3)
    plt.savefig("noise_image.png", dpi=200, bbox_inches='tight')
    plt.show()
    pass
