import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')

if __name__ == "__main__":
    receptive_node = [10, 12, 39, 45, 126, 144, 387, 441, 603, 2592]

    fig, axs = plt.subplots()
    t = [i for i in range(10)]
    axs.plot(t, receptive_node, marker='o', color='green')
    axs.set_xlim(0, 9)
    axs.set_ylim(0, 3000)
    labels = ['layer1 conv', 'layer1 pool', 'layer2 conv', 'layer2 pool', 'layer3 conv', 'layer3 pool', 'layer4 conv',
              'layer4 pool', 'layer5 conv', 'layer5 fc']
    plt.xticks(t, labels)
    plt.xticks(rotation=30)
    # axs.set_xlabel('layer')
    axs.set_ylabel('Receptive Field')

    # 设置数字标签
    for a, b in zip(t, receptive_node):
        if a == 0:
            plt.text(a + 0.2, b + 8, b, ha='center', va='bottom', fontsize=10)
        elif a == 9:
            plt.text(a - 0.4, b + 8, b, ha='center', va='bottom', fontsize=10)
        else:
            plt.text(a, b + 8, b, ha='center', va='bottom', fontsize=10)
    # plt.fill([0] * 10, receptive_node, 'b', alpha=0.5)
    plt.fill_between(t, receptive_node, interpolate=True, color='red', alpha=0.3)
    # axs.grid(True)
    plt.title("Receptive Field Growth")
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    plt.savefig("receptive_image.png", dpi=200, bbox_inches='tight')
    plt.show()
