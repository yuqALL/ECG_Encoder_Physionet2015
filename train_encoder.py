import logging
import sys
import numpy as np

import torch.nn.functional as F
from torch import optim
from torch_ext.util import set_random_seeds

from experiments.experiment_encoder import Experiment
from experiments.stopcriteria import MaxEpochs, Or, NoDecrease
from experiments.monitors import LossMonitor, RuntimeMonitor

from datasets.wfdb_dataset import DatasetLoader

from models.UnetEncode import UnetEncoder
from models.ssim import SSIM
from options.option_factory import gen_options

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def run_exp_on_wfdb_dataset(opt, exp_name='encoder'):
    dataset = DatasetLoader(opt)

    set_random_seeds(opt.np_to_seed, cuda=opt.cuda)

    if opt.debug:
        opt.max_epochs = 4
    result = []

    exp = single_exp(dataset, opt, exp_name)
    log.info("Last 10 epochs mean value")
    log.info("\n" + str(exp.epochs_df.iloc[-10:].mean()))
    t = np.array(exp.epochs_df.iloc[-10:].mean())
    result.append(t)
    print(result)


def single_exp(dataset, opt, exp_name):
    model = UnetEncoder(inchans=1)
    if opt.cuda:
        model.cuda()
    print(model)
    model.eval()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), weight_decay=opt.weight_decay,
                           lr=opt.lr)
    monitors = [RuntimeMonitor(), LossMonitor()]

    # model_constraint = None

    def testloss(outputs, inputs):
        print(outputs)
        print(inputs)
        print(outputs.size(), inputs.size())
        return F.mse_loss(outputs, inputs, reduction='mean')

    # ssim = SSIM()
    loss_function = lambda outputs, inputs: F.mse_loss(outputs, inputs, reduction='mean')

    # model_loss = lambda outputs, inputs: 1 - ssim(inputs, outputs)
    do_early_stop = True

    stop_criterion = Or([MaxEpochs(opt.max_epoch),
                         NoDecrease('train_loss', opt.max_increase_epoch)])
    # stop_criterion = None
    remember_best_column = 'test_loss'
    path = './checkpoints/encoder_'

    exp = Experiment(opt, model, datasets=dataset,
                     loss_function=loss_function, optimizer=optimizer,
                     monitors=monitors,
                     model_constraint=None,
                     stop_criterion=stop_criterion,
                     model_loss_function=None,
                     remember_best_column=remember_best_column,
                     do_early_stop=do_early_stop,
                     training_increase=False,
                     save_path=path + exp_name)
    exp.run()
    return exp


if __name__ == '__main__':
    # 设置Log格式
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)

    opt = gen_options(use_norm=True, use_noise=True)
    run_exp_on_wfdb_dataset(opt, exp_name=opt.get_name())

    sys.exit(0)
