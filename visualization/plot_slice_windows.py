from datasets.data_read_utils import load_record
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.patches as mpatches
from sys import exit

plt.rc('font', family='Times New Roman')
if __name__ == "__main__":
    sig1 = load_record('./a103l')
    sig2 = load_record('./a109l')
    eeg = sig1.p_signal[:, 0].T[71250:75000]
    pleth = sig1.p_signal[:, 2].T[71250:75000]
    abp = sig2.p_signal[:, 2].T[71250:75000]
    resp = sig2.p_signal[:, 3].T[71250:75000]

    y_ticks = ['mV', 'NU', 'mmHg', 'NU']

    t = [i for i in range(3750)]

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(5, 6)
    fig.align_labels()
    axs[0].plot(t, eeg, color='black')
    axs[0].set_xlim(0, 3750)
    axs[0].set_title('ECG', fontsize=12)
    axs[0].set_ylabel(y_ticks[0])
    # axs[0].grid(True)
    labels = ['4:45', '4:50', '4:55', '5:00']
    # axs[0].xticks([0, 1250, 2500, 3750], labels)
    # axs[0].set_xticks([0, 1250, 2500, 3750])
    # axs[0].set_xticklabels(labels)
    # plt.xticks(rotation=30)
    miny = min(eeg)
    maxy = max(eeg)
    height = maxy - miny
    left_down = np.array([250, miny])
    # 长方形
    rect = mpathes.Rectangle(left_down, 3000, (maxy - miny) * 0.95,
                             color='darkgray',
                             alpha=0.8,
                             ls='--',
                             lw=2,
                             fill=True)
    axs[0].add_patch(rect)
    left_down = np.array([375, miny])
    rect2 = mpathes.Rectangle((375, miny + height * 0.05), 3000, (maxy - miny) * 0.95,
                              color='crimson',
                              alpha=0.5,
                              ls='--',
                              lw=2,
                              fill=True)
    axs[0].add_patch(rect2)

    axs[1].plot(t, pleth, color='black')
    axs[1].set_xlim(0, 3750)
    # axs[1].set_xlabel('time')
    axs[1].set_title('PPG', fontsize=12)
    axs[1].set_ylabel(y_ticks[1])
    # axs[1].set_xticks([0, 1250, 2500, 3750])
    # axs[1].set_xticklabels(labels)

    miny = min(pleth)
    maxy = max(pleth)
    height = maxy - miny
    left_down = np.array([250, miny])
    # 长方形
    rect = mpathes.Rectangle(left_down, 3000, (maxy - miny) * 0.95,
                             color='darkgray',
                             alpha=0.8,
                             ls='--',
                             lw=2,
                             fill=True)
    axs[1].add_patch(rect)
    left_down = np.array([375, miny])
    rect2 = mpathes.Rectangle((375, miny + height * 0.05), 3000, (maxy - miny) * 0.95,
                              color='crimson',
                              alpha=0.5,
                              ls='--',
                              lw=2,
                              fill=True)
    axs[1].add_patch(rect2)

    axs[2].plot(t, abp, color='black')
    axs[2].set_xlim(0, 3750)
    # axs[2].set_xlabel('time')
    axs[2].set_title('ABP', fontsize=12)
    axs[2].set_ylabel(y_ticks[2])
    # axs[2].set_xticks([0, 1250, 2500, 3750])
    # axs[2].set_xticklabels(labels)

    miny = min(abp)
    maxy = max(abp)
    height = maxy - miny
    left_down = np.array([250, miny])
    # 长方形
    rect = mpathes.Rectangle(left_down, 3000, (maxy - miny) * 0.95,
                             color='darkgray',
                             alpha=0.8,
                             ls='--',
                             lw=2,
                             fill=True)
    axs[2].add_patch(rect)
    left_down = np.array([375, miny])
    rect2 = mpathes.Rectangle((375, miny + height * 0.05), 3000, (maxy - miny) * 0.95,
                              color='crimson',
                              alpha=0.5,
                              ls='--',
                              lw=2,
                              fill=True)
    axs[2].add_patch(rect2)

    axs[3].plot(t, resp, color='black')
    axs[3].set_xlim(0, 3750)
    axs[3].set_xlabel('time')
    axs[3].set_title('RESP')
    axs[3].set_ylabel(y_ticks[3])
    axs[3].set_xticks([0, 1250, 2500, 3750])
    axs[3].set_xticklabels(labels)

    miny = min(resp)
    maxy = max(resp)
    height = maxy - miny
    left_down = np.array([250, miny])
    # 长方形
    rect = mpathes.Rectangle(left_down, 3000, (maxy - miny) * 0.95,
                             color='darkgray',
                             alpha=0.8,
                             ls='--',
                             lw=2,
                             fill=True)
    axs[3].add_patch(rect)
    left_down = np.array([375, miny])
    rect2 = mpathes.Rectangle((375, miny + height * 0.05), 3000, (maxy - miny) * 0.95,
                              color='crimson',
                              alpha=0.5,
                              ls='--',
                              lw=2,
                              fill=True)
    axs[3].add_patch(rect2)

    fig.tight_layout()
    color = ['darkgray', 'crimson']
    # 我试过这里不能直接生成legend，解决方法就是自己定义，创建legend
    labels = ['Nth slice', '(N+1)th slice']  # legend标签列表，上面的color即是颜色列表
    # 用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    patches = [mpatches.Patch(color=color[i], alpha=0.5, label="{:s}".format(labels[i])) for i in range(len(color))]

    ax = plt.gca()
    box = ax.get_position()
    # print(box)
    # ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # 下面一行中bbox_to_anchor指定了legend的位置
    plt.legend(handles=patches, bbox_to_anchor=(box.width + 0.2, -0.3), ncol=1)  # 生成legend
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.3)
    plt.savefig("slice_windows_image.png", dpi=200, bbox_inches='tight')
    plt.show()
    # exit(0)
