import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from torch_ext.modules import Expression, AvgPool2dWithConv
from torch_ext.functions import identity
from torch_ext.util import np_to_var
from .util import initialize_weights
from options.options_setting import Opt


# # writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
# # from tensorboardX import SummaryWriter
# #
# # writer = SummaryWriter('log')  # 建立一个保存数据用的东西
# # import tensorwatch as tw

class DeeperNet(nn.Module):

    def __init__(
            self,
            opt=None
    ):
        super(DeeperNet, self).__init__()
        self.opt = opt

        conv_chans = 8 * self.opt.input_nc

        self.conv_time1 = self.base_layer(opt.input_nc, conv_chans, opt.input_nc)
        self.conv_time2 = self.base_layer(conv_chans, 2 * conv_chans, opt.input_nc)
        self.conv_time3 = self.base_layer(2 * conv_chans, 2 * conv_chans, opt.input_nc)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(p=0.3)

        self.block21 = self.base_layer(2 * conv_chans, 2 * conv_chans, opt.input_nc)
        self.block22 = self.base_layer(2 * conv_chans, 4 * conv_chans, opt.input_nc)
        self.block23 = self.base_layer(4 * conv_chans, 4 * conv_chans, opt.input_nc)

        self.block31 = self.base_layer(4 * conv_chans, 4 * conv_chans, opt.input_nc)
        self.block32 = self.base_layer(4 * conv_chans, 8 * conv_chans, opt.input_nc)
        self.block33 = self.base_layer(8 * conv_chans, 8 * conv_chans, opt.input_nc)

        self.block41 = self.base_layer(8 * conv_chans, 8 * conv_chans, opt.input_nc)
        self.block42 = self.base_layer(8 * conv_chans, 16 * conv_chans, opt.input_nc)
        self.block43 = self.base_layer(16 * conv_chans, 8 * conv_chans, opt.input_nc)

        self.conv_channel_res = self.base_layer(8 * conv_chans, opt.input_nc * 4, groups=1)
        out = self.test(
            np_to_var(
                np.ones(
                    (1, self.opt.input_nc, self.opt.input_length),
                    dtype=np.float32,
                )
            )
        )
        n_out_time = out.cpu().data.numpy().shape[2]
        self.final_conv_length = n_out_time

        self.conv_classifier = nn.Conv1d(
            in_channels=opt.input_nc * 4,
            out_channels=self.opt.n_classes,
            kernel_size=self.final_conv_length,
            bias=True,
        )

        self.softmax = nn.Softmax(dim=1)
        initialize_weights(self)

    def base_layer(self, inchans, outchans, groups):
        model = []
        model += [nn.Conv1d(inchans, outchans, kernel_size=3, stride=1, groups=groups),
                  nn.BatchNorm1d(outchans),
                  nn.LeakyReLU(0.2, inplace=True)]
        net = nn.Sequential(*model)
        initialize_weights(net)
        return net

    def test(self, x):
        print(x.shape)
        x = self.conv_time3(self.conv_time2(self.conv_time1(x)))
        x = self.dropout(self.pool(x))
        x = self.block23(self.block22(self.block21(x)))
        x = self.dropout(self.pool(x))
        x = self.block33(self.block32(self.block31(x)))
        x = self.dropout(self.pool(x))
        x = self.block43(self.block42(self.block41(x)))
        x = self.conv_channel_res(x)
        return x

    def forward(self, x):
        x = self.conv_time3(self.conv_time2(self.conv_time1(x)))
        x = self.dropout(self.pool(x))
        x = self.block23(self.block22(self.block21(x)))
        x = self.dropout(self.pool(x))
        x = self.block33(self.block32(self.block31(x)))
        x = self.dropout(self.pool(x))
        x = self.block43(self.block42(self.block41(x)))
        x = self.conv_channel_res(x)
        x = self.conv_classifier(x)
        x = self.softmax(x)
        return x

    def predict(self, input):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input)
            outputs = outputs.cpu().detach().numpy()
        print(outputs)
        if outputs[0, 0, 0] >= 0.5:
            return 1
        return 0


if __name__ == "__main__":
    from torchsummary import summary

    input = np_to_var(
        np.ones(
            (1, 5, 2500),
            dtype=np.float32,
        )
    )
    model = DeepMNet(5, 2, input_time_length=2502,
                     n_filters_time=8,
                     final_conv_length="auto", drop_prob=0.8).cpu()
    # img_model=tw.draw_model(model, [20, 5, 2502],orientation='LR')
    # img_model.save("ppp.png")
    # img_model.save(r'L:/backup/Ecgdecode/model_p.png')
    # model = DeepMNet(5, 2, input_time_length=2502,
    #                  n_filters_time=25,
    #                  final_conv_length="auto", drop_prob=0.8).cpu()
    # img_param=tw.model_stats(model, [20, 5, 2502])
    # img_param.save('model_param.png')
    # dummy_input = torch.rand(20, 5, 2502)  # 假设输入20张1*28*28的图片
    #
    # with SummaryWriter(comment='DeepNetM') as w:
    #     w.add_graph(model, (dummy_input,))
    summary(model, (5, 2500))
