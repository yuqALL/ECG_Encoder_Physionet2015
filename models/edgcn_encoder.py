import os
import numpy as np
import torch
from torch import nn
from torch_ext.util import np_to_var
from models.util import initialize_weights
from options.options_setting import Opt
from torch_ext.modules import Expression
from models.UnetEncode import UnetEncoder


class EDGCNEncoder(nn.Module):

    def __init__(
            self,
            opt=None
    ):
        super(EDGCNEncoder, self).__init__()
        self.opt = opt
        self.conv_stride = 1
        self.pool_stride = 3
        encoder_block = [self.load_feature_model(UnetEncoder(inchans=1), self.opt.encoder_model_path),
                         self.add_base_block(1, 1, 3, 10, use_dropout=True),
                         self.add_base_block(self.opt.input_nc, 1, 3, 11, use_group=False, use_dropout=True)
                         ]
        self.encoder = nn.Sequential(*encoder_block)

        block = []
        block += self.add_base_block(1, self.opt.n_filters, self.opt.filter_length, 1)
        block += self.add_base_block(
            self.opt.n_filters, self.opt.n_filters_2, self.opt.filter_length_2, 2, use_dropout=True
        )
        block += self.add_base_block(
            self.opt.n_filters_2, self.opt.n_filters_3, self.opt.filter_length_3, 3
        )
        block += self.add_base_block(
            self.opt.n_filters_3, self.opt.n_filters_4, self.opt.filter_length_4, 4
        )
        self.pre_layers = nn.Sequential(*block)
        self.block_decision = self.add_decision_block(self.encoder, self.pre_layers)
        initialize_weights(self)

    def add_base_block(self, n_filters_before, n_filters, filter_length, block_nr, use_group=True, use_dropout=False):
        if use_group:
            n_filters_before = n_filters_before * self.opt.input_nc
            n_filters = n_filters * self.opt.input_nc
        model = nn.Sequential()
        suffix = "_{:d}".format(block_nr)
        if use_dropout:
            model.add_module("drop" + suffix, nn.Dropout(p=self.opt.drop_prob))
        model.add_module(
            "conv" + suffix,
            nn.Conv1d(
                n_filters_before,
                n_filters,
                filter_length,
                stride=self.conv_stride,
                groups=self.opt.input_nc if use_group else 1,
                bias=not self.opt.batch_norm,
            ),
        )
        if self.opt.batch_norm:
            model.add_module("bnorm" + suffix,
                             nn.BatchNorm1d(n_filters, momentum=0.1, affine=True, eps=1e-5))

        model.add_module("nonlin" + suffix, nn.LeakyReLU(0.2, inplace=True))
        model.add_module("pool" + suffix, nn.MaxPool1d(kernel_size=3, stride=self.pool_stride))
        return model

    def concat_extra(self, x, extra, encode):
        B, C, W = x.size()
        x = torch.cat((x.view(B, -1), encode.reshape(B, -1), extra), dim=1).unsqueeze(dim=1)
        return x

    def flaten_x(self, x):
        B, C, W = x.size()
        x = x.view(B, -1)
        return x

    def add_decision_block(self, encoder, pre_layers):
        model = nn.Sequential()
        suffix = "_decision_block"
        model.add_module("conv" + suffix, nn.Conv1d(
            in_channels=self.opt.n_filters_4 * self.opt.input_nc,
            out_channels=self.opt.input_nc,
            kernel_size=self.opt.channel_res_conv_length,
            bias=True,
        ))
        model.add_module("nonlin" + suffix, nn.ELU())
        tmp = torch.ones((1, self.opt.input_nc, self.opt.input_length), dtype=torch.float32, requires_grad=False)
        n_out_time = pre_layers(tmp).cpu().data.numpy().shape[2] - 2
        n_out_encoder = encoder(tmp).cpu().data.numpy().shape[2]
        model.add_module("reshape" + suffix, Expression(self.concat_extra))
        model.add_module("classifier_conv" + suffix, nn.Conv1d(1, 32,
                                                               kernel_size=self.opt.input_nc *
                                                                           n_out_time + self.opt.extra_length + n_out_encoder))
        model.add_module("drop" + suffix, nn.Dropout(self.opt.drop_prob))
        model.add_module("view" + suffix, Expression(self.flaten_x))
        model.add_module("fc" + suffix, nn.Linear(32, 2, bias=True))
        model.add_module("logSoftmax" + suffix, nn.LogSoftmax(dim=1))

        def forward(x, extra, encode):
            for name, module in model._modules.items():
                if name != 'reshape' + suffix:
                    x = module(x)
                else:
                    x = module(x, extra, encode)
            return x

        model.forward = forward
        return model

    def load_feature_model(self, model, path):
        if not os.path.exists(path):
            return None

        model.load_state_dict(torch.load(path))
        # if opt.cuda:
        #     model.cuda()
        if model is None:
            return None

        def forward(x):
            B, C, W = x.size()
            all_features = []
            for i in range(C):
                tx = model.down(model.lrelu(model.conv_first(x[:, [i], :])))
                all_features.append(torch.flatten(tx, 1, 2))
            x = torch.stack(all_features)
            x = x.transpose(0, 1)
            return x

        model.forward = forward

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def forward(self, x, extra):
        # print(x.size())
        encode = self.encoder(x)
        x = self.pre_layers(x)
        x = self.block_decision(x, extra, encode)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    opt = Opt()
    opt.input_nc = 2
    opt.input_length = 3072
    opt.cuda = False
    opt.extra_length = 5
    opt.encoder_model_path = "../checkpoints/encoder_1.pth"
    model = EDGCNEncoder(opt).cpu()
    summary(model, [(2, 3072), (5,)], device='cpu')
