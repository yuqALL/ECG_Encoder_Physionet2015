from datasets.data_read_utils import load_record
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.patches as mpatches
from sys import exit
import copy

plt.rc('font', family='Times New Roman')


def gaussion_noise(sig, sigma='default'):
    cnt = copy.deepcopy(sig)
    if sigma == 'default':
        sigma = 0.01 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    cnt += noise
    return cnt


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
    eeg = sig1.p_signal[:, 0].T[73750:75000]
    pleth = sig1.p_signal[:, 2].T[73750:75000]
    abp = sig2.p_signal[:, 2].T[73750:75000]
    resp = sig2.p_signal[:, 3].T[73750:75000]

    y_ticks = ['mV', 'NU', 'mmHg', 'NU']

    t = [i for i in range(1250)]

    fig, axs = plt.subplots(2, 2)
    # eeg = eeg[:, np.newaxis]
    # pleth = pleth[:, np.newaxis]
    # abp = abp[:, np.newaxis]
    # resp = resp[:, np.newaxis]

    # axs[0][0].plot(t, sin_gaussion_noise(eeg), color='red', linestyle='-', label='nosing signal')
    axs[0][0].plot(t, eeg, color='red', label='origin signal')
    axs[0][0].set_xlim(0, 1250)
    axs[0][0].set_xlabel('time')
    axs[0][0].set_title('EEG Signal', fontsize=12)
    axs[0][0].set_ylabel(y_ticks[0])
    labels = ['4:55', '4:51', '4:52', '4:53', '4:54', '5:00']
    axs[0][0].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[0][0].set_xticklabels(labels)
    # axs[0][0].legend()

    # axs[0][1].plot(t, sin_gaussion_noise(pleth), color='red', linestyle='-', label='nosing signal')
    axs[0][1].plot(t, pleth, color='green', label='origin signal')
    axs[0][1].set_xlim(0, 1250)
    axs[0][1].set_xlabel('time')
    axs[0][1].set_title('PPG Signal', fontsize=12)
    axs[0][1].set_ylabel(y_ticks[1])
    axs[0][1].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[0][1].set_xticklabels(labels)
    # axs[0][1].legend()

    # axs[1][0].plot(t, sin_gaussion_noise(abp), color='red', linestyle='-', label='nosing signal')
    axs[1][0].plot(t, abp, color='blue', label='origin signal')
    axs[1][0].set_xlim(0, 1250)
    axs[1][0].set_xlabel('time')
    axs[1][0].set_title('ABP Signal', fontsize=12)
    axs[1][0].set_ylabel(y_ticks[2])
    axs[1][0].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[1][0].set_xticklabels(labels)
    # axs[1][0].legend()

    # axs[1][1].plot(t, sin_gaussion_noise(resp), color='red', linestyle='-', label='nosing signal')
    axs[1][1].plot(t, resp, color='black', label='origin signal')
    axs[1][1].set_xlim(0, 1250)
    axs[1][1].set_xlabel('time')
    axs[1][1].set_title('RESP Signal', fontsize=12)
    axs[1][1].set_ylabel(y_ticks[3])
    axs[1][1].set_xticks([0, 250, 500, 750, 1000, 1250])
    axs[1][1].set_xticklabels(labels)
    # axs[1][1].legend()

    fig.tight_layout()

    plt.show()
    pass
