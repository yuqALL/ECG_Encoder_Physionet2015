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


def read_data(path, model_name='DGCN'):
    full_data = pd.read_csv(os.path.join(path, "{}_3750.csv".format(model_name)))
    slice_data = pd.read_csv(os.path.join(path, "{}_3000.csv".format(model_name)))
    slice_data_norm = pd.read_csv(os.path.join(path, "{}_3000_norm.csv".format(model_name)))
    slice_data_noise = pd.read_csv(os.path.join(path, "{}_3000_noise.csv".format(model_name)))
    slice_data_norm_noise = pd.read_csv(
        os.path.join(path, "{}_3000_norm_noise.csv".format(model_name)))

    return full_data.values[:201] / 100, slice_data.values[:201] / 100, \
           slice_data_norm.values[:201] / 100, slice_data_noise.values[:201] / 100, \
           slice_data_norm_noise.values[:201] / 100


def get_data_min_max(data1, data2, data3, data4, data5):
    data_min = min(min(data1), min(data2), min(data3), min(data4),
                   min(data5))
    data_max = max(max(data1), max(data2), max(data3), max(data4),
                   max(data5))
    return data_min, data_max


def plot_model(data_path="./"):
    dgcn_15s, dgcn_12s, dgcn_12s_norm, dgcn_12s_noise, dgcn_12s_norm_noise = read_data(data_path, model_name='DGCN')
    edgcn_15s, edgcn_12s, edgcn_12s_norm, edgcn_12s_noise, edgcn_12s_norm_noise = read_data(data_path,
                                                                                            model_name='EDGCN')
    combined_edgcn_15s, combined_edgcn_12s, combined_edgcn_12s_norm, combined_edgcn_12s_noise, combined_edgcn_12s_norm_noise = read_data(
        data_path,
        model_name='Combined')
    dgcn_acc_min, dgcn_acc_max = get_data_min_max(dgcn_15s[:, 8], dgcn_12s[:, 8], dgcn_12s_norm[:, 8],
                                                  dgcn_12s_noise[:, 8],
                                                  dgcn_12s_norm_noise[:, 8])
    dgcn_auc_min, dgcn_auc_max = get_data_min_max(dgcn_15s[:, 9], dgcn_12s[:, 9], dgcn_12s_norm[:, 9],
                                                  dgcn_12s_noise[:, 9],
                                                  dgcn_12s_norm_noise[:, 9])
    dgcn_score_min, dgcn_score_max = get_data_min_max(dgcn_15s[:, 10], dgcn_12s[:, 10], dgcn_12s_norm[:, 10],
                                                      dgcn_12s_noise[:, 10], dgcn_12s_norm_noise[:, 10])

    edgcn_acc_min, edgcn_acc_max = get_data_min_max(edgcn_15s[:, 8], edgcn_12s[:, 8], edgcn_12s_norm[:, 8],
                                                    edgcn_12s_noise[:, 8],
                                                    edgcn_12s_norm_noise[:, 8])
    edgcn_auc_min, edgcn_auc_max = get_data_min_max(edgcn_15s[:, 9], edgcn_12s[:, 9], edgcn_12s_norm[:, 9],
                                                    edgcn_12s_noise[:, 9],
                                                    edgcn_12s_norm_noise[:, 9])
    edgcn_score_min, edgcn_score_max = get_data_min_max(edgcn_15s[:, 10], edgcn_12s[:, 10], edgcn_12s_norm[:, 10],
                                                        edgcn_12s_noise[:, 10],
                                                        edgcn_12s_norm_noise[:, 10])

    combined_edgcn_acc_min, combined_edgcn_acc_max = get_data_min_max(combined_edgcn_15s[:, 8],
                                                                      combined_edgcn_12s[:, 8],
                                                                      combined_edgcn_12s_norm[:, 8],
                                                                      combined_edgcn_12s_noise[:, 8],
                                                                      combined_edgcn_12s_norm_noise[:, 8])
    combined_edgcn_auc_min, combined_edgcn_auc_max = get_data_min_max(combined_edgcn_15s[:, 9],
                                                                      combined_edgcn_12s[:, 9],
                                                                      combined_edgcn_12s_norm[:, 9],
                                                                      combined_edgcn_12s_noise[:, 9],
                                                                      combined_edgcn_12s_norm_noise[:, 9])
    combined_edgcn_score_min, combined_edgcn_score_max = get_data_min_max(combined_edgcn_15s[:, 10],
                                                                          combined_edgcn_12s[:, 10],
                                                                          combined_edgcn_12s_norm[:, 10],
                                                                          combined_edgcn_12s_noise[:, 10],
                                                                          combined_edgcn_12s_norm_noise[:, 10])

    y0_min = min(dgcn_acc_min, edgcn_acc_min, combined_edgcn_acc_min)
    y0_max = max(dgcn_acc_max, edgcn_acc_max, combined_edgcn_acc_max)

    y1_min = min(dgcn_auc_min, edgcn_auc_min, combined_edgcn_auc_min)
    y1_max = max(dgcn_auc_max, edgcn_auc_max, combined_edgcn_auc_max)

    y2_min = min(dgcn_score_min, edgcn_score_min, combined_edgcn_score_min)
    y2_max = max(dgcn_score_max, edgcn_score_max, combined_edgcn_score_max)

    y_data1 = meanSubWin(dgcn_15s[:, 8:], 3)
    y_data2 = meanSubWin(dgcn_12s[:, 8:], 3)
    y_data3 = meanSubWin(dgcn_12s_norm[:, 8:], 3)
    y_data4 = meanSubWin(dgcn_12s_noise[:, 8:], 3)
    y_data5 = meanSubWin(dgcn_12s_norm_noise[:, 8:], 3)

    # 开始绘图
    fig, ax1 = plt.subplots(3, 3, sharey='row', sharex='col')
    fig.set_size_inches(12, 8)
    plt.set_cmap('RdBu')
    # multiple line plot

    lw = 2  # 控制线条的粗细

    x = np.arange(start=0, stop=max(len(y_data1), len(y_data2), len(y_data3), len(y_data4), len(y_data5)),
                  step=50)  # x坐标的范围
    a, = ax1[0][0].plot(np.arange(0, len(y_data1)), y_data1[:, 0], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[0][0].plot(np.arange(0, len(y_data2)), y_data2[:, 0], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[0][0].plot(np.arange(0, len(y_data3)), y_data3[:, 0], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[0][0].plot(np.arange(0, len(y_data4)), y_data4[:, 0], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[0][0].plot(np.arange(0, len(y_data5)), y_data5[:, 0], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[0][0].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[0][0].set_xticks(x)  # 设置xticks出现的位置
    ax1[0][0].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    ax1[0][0].set_ylabel("{} value".format('ACC'), fontsize='xx-large')
    # 设置标题
    ax1[0][0].set_title('DGCN', fontsize='x-large')
    ax1[0][0].legend(loc='lower right')

    a, = ax1[1][0].plot(np.arange(0, len(y_data1)), y_data1[:, 1], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[1][0].plot(np.arange(0, len(y_data2)), y_data2[:, 1], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[1][0].plot(np.arange(0, len(y_data3)), y_data3[:, 1], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[1][0].plot(np.arange(0, len(y_data4)), y_data4[:, 1], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[1][0].plot(np.arange(0, len(y_data5)), y_data5[:, 1], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[1][0].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[1][0].set_xticks(x)  # 设置xticks出现的位置
    ax1[1][0].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    ax1[1][0].set_ylabel("{} value".format('AUC'), fontsize='xx-large')
    ax1[1][0].legend(loc='lower right')

    a, = ax1[2][0].plot(np.arange(0, len(y_data1)), y_data1[:, 2], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[2][0].plot(np.arange(0, len(y_data2)), y_data2[:, 2], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[2][0].plot(np.arange(0, len(y_data3)), y_data3[:, 2], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[2][0].plot(np.arange(0, len(y_data4)), y_data4[:, 2], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[2][0].plot(np.arange(0, len(y_data5)), y_data5[:, 2], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[2][0].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[2][0].set_xticks(x)  # 设置xticks出现的位置
    ax1[2][0].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    ax1[2][0].set_ylabel("{} value".format('Score'), fontsize='xx-large')
    ax1[2][0].set_xlabel("epoch", fontsize='xx-large')
    ax1[2][0].legend(loc='lower right')

    # plt.legend(handles=[a, b, c, d, e], loc='lower right')

    y_data1 = meanSubWin(edgcn_15s[:, 8:], 3)
    y_data2 = meanSubWin(edgcn_12s[:, 8:], 3)
    y_data3 = meanSubWin(edgcn_12s_norm[:, 8:], 3)
    y_data4 = meanSubWin(edgcn_12s_noise[:, 8:], 3)
    y_data5 = meanSubWin(edgcn_12s_norm_noise[:, 8:], 3)

    x = np.arange(start=0, stop=max(len(y_data1), len(y_data2), len(y_data3), len(y_data4), len(y_data5)),
                  step=50)  # x坐标的范围
    a, = ax1[0][1].plot(np.arange(0, len(y_data1)), y_data1[:, 0], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[0][1].plot(np.arange(0, len(y_data2)), y_data2[:, 0], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[0][1].plot(np.arange(0, len(y_data3)), y_data3[:, 0], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[0][1].plot(np.arange(0, len(y_data4)), y_data4[:, 0], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[0][1].plot(np.arange(0, len(y_data5)), y_data5[:, 0], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[0][1].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[0][1].set_xticks(x)  # 设置xticks出现的位置
    ax1[0][1].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    # ax1[0][1].set_ylabel("{} value".format('ACC'), fontsize='xx-large')
    # 设置标题
    ax1[0][1].set_title('EDGCN', fontsize='x-large')
    ax1[0][1].legend(loc='lower right')

    a, = ax1[1][1].plot(np.arange(0, len(y_data1)), y_data1[:, 1], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[1][1].plot(np.arange(0, len(y_data2)), y_data2[:, 1], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[1][1].plot(np.arange(0, len(y_data3)), y_data3[:, 1], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[1][1].plot(np.arange(0, len(y_data4)), y_data4[:, 1], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[1][1].plot(np.arange(0, len(y_data5)), y_data5[:, 1], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[1][1].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[1][1].set_xticks(x)  # 设置xticks出现的位置
    ax1[1][1].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    # ax1[1][1].set_ylabel("{} value".format('AUC'), fontsize='xx-large')

    ax1[1][1].legend(loc='lower right')

    a, = ax1[2][1].plot(np.arange(0, len(y_data1)), y_data1[:, 2], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[2][1].plot(np.arange(0, len(y_data2)), y_data2[:, 2], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[2][1].plot(np.arange(0, len(y_data3)), y_data3[:, 2], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[2][1].plot(np.arange(0, len(y_data4)), y_data4[:, 2], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[2][1].plot(np.arange(0, len(y_data5)), y_data5[:, 2], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[2][1].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[2][1].set_xticks(x)  # 设置xticks出现的位置
    ax1[2][1].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    # ax1[2][1].set_ylabel("{} value".format('Score'), fontsize='xx-large')
    ax1[2][1].set_xlabel("epoch", fontsize='xx-large')

    ax1[2][1].legend(loc='lower right')

    y_data1 = meanSubWin(combined_edgcn_15s[:, 8:], 3)
    y_data2 = meanSubWin(combined_edgcn_12s[:, 8:], 3)
    y_data3 = meanSubWin(combined_edgcn_12s_norm[:, 8:], 3)
    y_data4 = meanSubWin(combined_edgcn_12s_noise[:, 8:], 3)
    y_data5 = meanSubWin(combined_edgcn_12s_norm_noise[:, 8:], 3)

    x = np.arange(start=0, stop=max(len(y_data1), len(y_data2), len(y_data3), len(y_data4), len(y_data5)),
                  step=50)  # x坐标的范围
    a, = ax1[0][2].plot(np.arange(0, len(y_data1)), y_data1[:, 0], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[0][2].plot(np.arange(0, len(y_data2)), y_data2[:, 0], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[0][2].plot(np.arange(0, len(y_data3)), y_data3[:, 0], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[0][2].plot(np.arange(0, len(y_data4)), y_data4[:, 0], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[0][2].plot(np.arange(0, len(y_data5)), y_data5[:, 0], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[0][2].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[0][2].set_xticks(x)  # 设置xticks出现的位置
    ax1[0][2].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    # ax1[0][2].set_ylabel("{} value".format('ACC'), fontsize='xx-large')
    # 设置标题
    ax1[0][2].set_title('Combined model', fontsize='x-large')
    ax1[0][2].legend(loc='lower right')

    a, = ax1[1][2].plot(np.arange(0, len(y_data1)), y_data1[:, 1], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[1][2].plot(np.arange(0, len(y_data2)), y_data2[:, 1], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[1][2].plot(np.arange(0, len(y_data3)), y_data3[:, 1], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[1][2].plot(np.arange(0, len(y_data4)), y_data4[:, 1], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[1][2].plot(np.arange(0, len(y_data5)), y_data5[:, 1], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[1][2].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[1][2].set_xticks(x)  # 设置xticks出现的位置
    ax1[1][2].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围
    # 设置坐标轴名称
    # ax1[1][2].set_ylabel("{} value".format('AUC'), fontsize='xx-large')

    ax1[1][2].legend(loc='lower right')

    a, = ax1[2][2].plot(np.arange(0, len(y_data1)), y_data1[:, 2], linewidth=lw, label='complete 15s signal',
                        # marker='o',
                        markersize=4)
    b, = ax1[2][2].plot(np.arange(0, len(y_data2)), y_data2[:, 2], linewidth=lw, label='12s slicing',
                        # marker='x',
                        markersize=4,
                        color='green')
    # c, = ax1[2][2].plot(np.arange(0, len(y_data3)), y_data3[:, 2], linewidth=lw, label='12s slicing with 0-1 normalization',
    #                     # marker='v',
    #                     markersize=4)
    d, = ax1[2][2].plot(np.arange(0, len(y_data4)), y_data4[:, 2], linewidth=lw,
                        label='12s slicing with noise addition',
                        # marker='*',
                        markersize=4,
                        color='crimson')
    # e, = ax1[2][2].plot(np.arange(0, len(y_data5)), y_data5[:, 2], linewidth=lw,
    #                     label='12s slicing with 0-1 normalization,noise addition',
    #                     # marker='^',
    #                     markersize=2)
    # 设置坐标轴的标签
    ax1[2][2].yaxis.set_tick_params(labelsize=15)  # 设置y轴的字体的大小
    ax1[2][2].set_xticks(x)  # 设置xticks出现的位置
    ax1[2][2].set_xlim(x[0], x[-1] + 10)  # 设置x轴的范围

    # 设置坐标轴名称
    # ax1[2][2].set_ylabel("{} value".format('Score'), fontsize='xx-large')
    ax1[2][2].set_xlabel("epoch", fontsize='xx-large')

    ax1[2][2].legend(loc='lower right')

    ax1[0][0].set_ylim(y0_min, y0_max)
    ax1[0][0].set_yticks(np.arange(start=int(y0_min * 10) / 10.0, stop=y0_max,
                                   step=0.1))
    ax1[1][0].set_ylim(y1_min, y1_max)
    ax1[1][0].set_yticks(np.arange(start=int(y1_min * 10) / 10.0, stop=y1_max,
                                   step=0.1))
    ax1[2][0].set_ylim(y2_min, y2_max)
    ax1[2][0].set_yticks(np.arange(start=int(y2_min * 10) / 10.0, stop=y2_max,
                                   step=0.1))

    # x_start = 0.4
    # x_end = 0.9
    # ax1[0][0].set_ylim(x_start, x_end)
    # ax1[0][0].set_yticks(np.arange(start=x_start, stop=x_end,
    #                                step=0.1))
    # ax1[1][0].set_ylim(x_start, x_end)
    # ax1[1][0].set_yticks(np.arange(start=x_start, stop=x_end,
    #                                step=0.1))
    # ax1[2][0].set_ylim(x_start, x_end)
    # ax1[2][0].set_yticks(np.arange(start=x_start, stop=x_end,
    #                                step=0.1))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.02, hspace=0.04)
    plt.savefig("图.png", dpi=200, bbox_inches='tight')
    plt.show()
    return


if __name__ == "__main__":
    # data1, data2, data3, data4, data5 = read_data('./', model_name='DGCN')
    plot_model()
