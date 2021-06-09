import os
from sys import exit
import logging
import numpy as np
import wfdb
import copy

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

r'''
本文件仅负责数据的读取
'''


def resample_sig(cnt, point=1000):
    from scipy import signal
    cnt_resample = signal.resample(cnt, point, axis=0)
    return cnt_resample


def fill_nan(signal):
    """Solution provided by Divakar."""
    mask = np.isnan(signal)
    idx = np.where(~mask.T, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = signal[idx.T, np.arange(idx.shape[0])[None, :]]
    out = np.nan_to_num(out)
    return out


def load_file_list(path, shuffle=True, seed=20200426):
    file_list = os.listdir(path)
    file_list = sorted(os.path.join(path, f[:-4]) for f in file_list if
                       os.path.isfile(os.path.join(path, f)) and f.endswith(".mat"))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(file_list)
    return file_list


def load_file(filename, data_path):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(os.path.join(data_path, line.strip('\n')))
    return content


def load_record(filename):
    record = wfdb.rdrecord(filename)
    return record


def load_sig_data(filename, fillnan=True):
    record = load_record(filename)
    fs = int(record.fs)
    cnt = np.full((fs * 300, len(record.sig_name)), np.nan, dtype='float32')
    continuous_signal = record.p_signal
    i = 0

    for j, s in enumerate(record.sig_name):
        cnt[:, i] = continuous_signal[0:fs * 300, j]
        i += 1

    if fillnan:
        cnt = fill_nan(cnt)
    return cnt


def minmax_scale(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    return cnt, minv, maxv


def minmax_scale_zero(cnt):
    minv = np.nanmin(cnt, axis=0)
    maxv = np.nanmax(cnt, axis=0)
    if isinstance(maxv, np.ndarray):
        t = [1 / v if v else 1 for v in (maxv - minv)]
    else:
        t = 1 / (maxv - minv) if maxv - minv else 1.0
    cnt -= minv
    cnt *= t
    cnt = (cnt - 0.5) * 2
    return cnt, minv, maxv


def gaussion_noise(sig, sigma='default', drop_noise=0.8):
    cnt = copy.deepcopy(sig)
    n, c = cnt.shape
    zeros = int(n * drop_noise)
    ones = n - zeros
    factor = np.array([0] * zeros + [1] * ones)

    if sigma == 'default':
        sigma = 0.1 * (np.nanmax(cnt, axis=0) - np.nanmin(cnt, axis=0))
        noise = sigma * np.random.randn(*cnt.shape)
    else:
        noise = sigma * np.random.randn(*cnt.shape)
    for i in range(c):
        np.random.shuffle(factor)
        noise[:, i] = factor * noise[:, i]
    cnt += noise
    return cnt, noise


def sub_window_split(cnt, test_mode=False, mmscale=False, add_noise_prob=0.0):
    start_point = 0
    end_point = 68126
    if test_mode:
        start_point = 71177
        end_point = 71928
    result_cnt = []
    origin_cnt = []
    ch = cnt.shape[1]
    for i in range(start_point, end_point, 125):
        info = np.zeros((3072, ch), dtype='float32')
        # if len(cnt[i:i + 3072]) != 3072:
        #     print(start_point, end_point, len(cnt))
        info[:] = cnt[i:i + 3072]
        if add_noise_prob > 0:
            origin_cnt.append(info)
        if mmscale:
            info, _, _ = minmax_scale_zero(info)

        if add_noise_prob > 0 and add_noise_prob > np.random.uniform():
            info, _ = gaussion_noise(info)

        result_cnt.append(info)
    if add_noise_prob > 0:
        return np.array(origin_cnt), np.array(result_cnt)
    return np.array(result_cnt)


if __name__ == "__main__":
    for i in range(0, 68250, 125):
        print(i, " -> ", i + 3072, "  <------>  ", i / 250, "s", " -> ", (i + 3072) / 250, "s")
    print()
    for i in range(300 * 250 - 1, 74248, -125):
        print(i - 3072, " -> ", i, "  <------>  ", (i - 3072) / 250, "s", " -> ", i / 250, "s")

    print()
    for i in range(71177, 71928, 125):
        print(i, " -> ", i + 3072, "  <------>  ", i / 250, "s", " -> ", (i + 3072) / 250, "s")
    exit(0)
