import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd
import os

plt.rc('font', family='Times New Roman')


def mean5_3(Series, m):
    n = len(Series)
    a = Series
    b = Series.copy()
    for i in range(m):
        b[0] = (69 * a[0] + 4 * (a[1] + a[3]) - 6 * a[2] - a[4]) / 70
        b[1] = (2 * (a[0] + a[4]) + 27 * a[1] + 12 * a[2] - 8 * a[3]) / 35
        for j in range(2, n - 2):
            b[j] = (-3 * (a[j - 2] + a[j + 2]) + 12 * (a[j - 1] + a[j + 1]) + 17 * a[j]) / 35
        b[n - 2] = (2 * (a[n - 1] + a[n - 5]) + 27 * a[n - 2] + 12 * a[n - 3] - 8 * a[n - 4]) / 35
        b[n - 1] = (69 * a[n - 1] + 4 * (a[n - 2] + a[n - 4]) - 6 * a[n - 3] - a[n - 5]) / 70
        a = b.copy()
    return a


def meanSubWin(Series, m=5):
    n = len(Series)
    a = Series
    b = Series.copy()
    x = []
    for i in range(m - 1, n):
        b[i] = sum(a[i - m + 1:i + 1]) / m
    return b


def read_data(path, prefix='ACC', model_name='DGCN'):
    full_data = pd.read_csv(os.path.join(path, "{}_{}_3750.csv".format(prefix, model_name)))
    slice_data = pd.read_csv(os.path.join(path, "{}_{}_3000.csv".format(prefix, model_name)))
    slice_data_norm = pd.read_csv(os.path.join(path, "{}_{}_3000_norm.csv".format(prefix, model_name)))
    slice_data_noise = pd.read_csv(os.path.join(path, "{}_{}_3000_noise.csv".format(prefix, model_name)))
    slice_data_norm_noise = pd.read_csv(
        os.path.join(path, "{}_{}_3000_norm_noise.csv".format(prefix, model_name)))

    return full_data.T, slice_data.T, slice_data_norm.T, slice_data_noise.T, slice_data_norm_noise.T


def plot_model(data_path="DGCN_DATA/", prefix='ACC', model_name='DGCN'):
    data1, data2, data3, data4, data5 = read_data(data_path, prefix=prefix, model_name=model_name)
    x = np.arange(start=0, stop=410, step=50)  # x坐标的范围
    s_data1 = interpolate.splrep(data1.loc["Step"].to_numpy(), data1.loc['Value'].to_numpy(), k=3, s=0.2)
    s_data2 = interpolate.splrep(data2.loc["Step"].to_numpy(), data2.loc['Value'].to_numpy(), k=3, s=0.2)
    s_data3 = interpolate.splrep(data3.loc["Step"].to_numpy(), data3.loc['Value'].to_numpy(), k=3, s=0.2)
    s_data4 = interpolate.splrep(data4.loc["Step"].to_numpy(), data4.loc['Value'].to_numpy(), k=3, s=0.2)
    s_data5 = interpolate.splrep(data5.loc["Step"].to_numpy(), data5.loc['Value'].to_numpy(), k=3, s=0.2)
    y_data1 = interpolate.splev(data1.loc["Step"].to_numpy(), s_data1, der=0)
    y_data2 = interpolate.splev(data2.loc["Step"].to_numpy(), s_data2, der=0)
    y_data3 = interpolate.splev(data3.loc["Step"].to_numpy(), s_data3, der=0)
    y_data4 = interpolate.splev(data4.loc["Step"].to_numpy(), s_data4, der=0)
    y_data5 = interpolate.splev(data5.loc["Step"].to_numpy(), s_data5, der=0)

    y_data1 = meanSubWin(data1.loc['Value'].to_numpy(), 10)
    y_data2 = meanSubWin(data2.loc['Value'].to_numpy(), 10)
    y_data3 = meanSubWin(data3.loc['Value'].to_numpy(), 10)
    y_data4 = meanSubWin(data4.loc['Value'].to_numpy(), 10)
    y_data5 = meanSubWin(data5.loc['Value'].to_numpy(), 10)

    # 开始绘图
    fig, ax1 = plt.subplots()
    fig.set_size_inches(7, 5)
    plt.set_cmap('RdBu')
    # multiple line plot

    lw = 2  # 控制线条的粗细
    a, = ax1.plot(data1.loc["Step"].to_numpy(), y_data1, linewidth=lw, label='All 15s signal',
                  # marker='o',
                  markersize=4)
    b, = ax1.plot(data2.loc['Step'].to_numpy(), y_data2, linewidth=lw, label='12s slice signal',
                  # marker='x',
                  markersize=4)
    c, = ax1.plot(data3.loc['Step'].to_numpy(), y_data3, linewidth=lw, label='12s slice,0-1 norm',
                  # marker='v',
                  markersize=4)
    d, = ax1.plot(data4.loc['Step'].to_numpy(), y_data4, linewidth=lw, label='12s slice,noising',
                  # marker='*',
                  markersize=4)
    e, = ax1.plot(data5.loc['Step'].to_numpy(), y_data5, linewidth=lw,
                  label='12s slice,0-1 norm,noising',
                  # marker='^',
                  markersize=2)

    plt.legend(handles=[a, b, c, d, e], loc='lower right')

    # 设置坐标轴的标签
    ax1.yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1.set_xticks(x)  # 设置xticks出现的位置
    ax1.set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    ax1.set_ylabel("{} value".format(prefix), fontsize='xx-large')
    ax1.set_xlabel("steps", fontsize='xx-large')
    # 设置标题
    ax1.set_title('Different models\' {} on our validation set'.format(prefix), fontsize='x-large')
    plt.show()


if __name__ == "__main__":
    # plot_dgcn(prefix='SCORE')
    plot_model(data_path="DGCN_DATA/", prefix='ACC', model_name='DGCN')
