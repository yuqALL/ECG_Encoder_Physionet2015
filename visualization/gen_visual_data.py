#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import sys
from sys import exit
import os
import numpy as np
import torch

from entry_bak.data_loader.data_read_utils import get_record_label
from models.model_loader import load_model as model_loader
import itertools
from datasets.data_loader import DataLoader
from datasets.wfdb_dataset import DatasetLoader
from options.options_setting import Opt
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import prettytable as pt
import logging
import math

log = logging.getLogger(__name__)
log.setLevel('DEBUG')
test_people_preds = list()
all_test_people_preds = defaultdict(list)
all_test_people_labels = defaultdict(int)
all_test_people_results = defaultdict(int)
all_test_people_poss = defaultdict(float)

file_dict = dict()
file_data = dict()


def load_model(path, opt):
    model = model_loader(opt)
    if not os.path.exists(path):
        return None
    if opt.cuda:
        model.cuda()
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_dataset(opt, alarm_type='all', explict_sensor=True):
    log.info("Loading dataset...")
    dataset = DatasetLoader(opt)
    dataset.filter_test_files(alarm_type, explict_sensor)
    return dataset


def vote():
    global all_test_people_preds, all_test_people_results, all_test_people_poss
    for f, preds in all_test_people_preds.items():
        tmp_preds = list(itertools.chain(*preds))
        n = len(tmp_preds)
        p = np.array(tmp_preds)[:, 0].sum() / n
        l = 1 if p < 0.5 else 0
        all_test_people_results[f] = l
        all_test_people_poss[f] = p

    return


def eval_on_single_people(inputs, extra, model, cuda=False):
    model.eval()
    with torch.no_grad():
        input_vars = torch.tensor(inputs, requires_grad=False)
        if np.any(extra):
            extra_vars = torch.tensor(extra, requires_grad=False)
            extra_vars = extra_vars.float()
            extra_vars = extra_vars.cuda() if cuda else extra_vars
        if cuda:
            input_vars = input_vars.cuda()

        outputs = model(input_vars) if not np.any(extra) else model(input_vars, extra_vars)
        outputs = torch.exp(outputs)
        if hasattr(outputs, "cpu"):
            outputs = outputs.cpu().detach().numpy()
        else:
            outputs = [o.cpu().detach().numpy() for o in outputs]
    return outputs


def init_data_dict(data_path):
    test_datapaths = ["1_fold_test_files_list.txt", "2_fold_test_files_list.txt", "3_fold_test_files_list.txt",
                      "4_fold_test_files_list.txt", "5_fold_test_files_list.txt"]
    for i, file_name in enumerate(test_datapaths):
        with open(data_path + file_name, 'r') as f:
            for line in f:
                file = os.path.join(data_path, line.strip('\n'))
                file_dict[file] = i


def predict(datasets, model, cuda=False):
    import time

    time_start = time.time()  # 开始计时

    # 要执行的代码，或函数
    # 要执行的代码，或函数

    global all_test_people_preds, all_test_people_labels

    for file in datasets.need_test_files:
        fold = file_dict[file]
        inputs, extra, targets = datasets.get_people(file, check=True)
        if not datasets.opt.use_extra:
            extra = None
        if np.all(inputs) is None:
            continue
        if isinstance(inputs, list):
            for sig in inputs:
                preds = eval_on_single_people(sig, extra, model[fold], cuda)
                all_test_people_preds[file].append(preds)
                all_test_people_labels[file] = targets[-1]
        else:
            preds = eval_on_single_people(inputs, extra, model[fold], cuda)
            all_test_people_preds[file].append(preds)
            all_test_people_labels[file] = targets[-1]
    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')
    return


all_dataset = defaultdict(DatasetLoader)


def init_load_data(opt, combined=True):
    if not combined:
        opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
        dataset1 = load_dataset(opt, 'all', False)
        all_dataset['ALL'] = dataset1
        return
    opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
    dataset1 = load_dataset(opt, 'all', False)
    all_dataset['PRA'] = dataset1

    opt.load_sensor_names = ['II']
    dataset2 = load_dataset(opt, 'all', False)
    all_dataset['II'] = dataset2

    opt.load_sensor_names = ['V']
    dataset3 = load_dataset(opt, 'all', False)
    all_dataset['V'] = dataset3

    opt.load_important_sig = True
    dataset4 = load_dataset(opt, 'all', False)
    opt.load_important_sig = False
    all_dataset['EEG'] = dataset4
    return


def predict_testfile(opt, alarm_type='all', explict_sensor=True, base_path=None,
                     model_name='best.pth', date_name='PRA', combined=True):
    # dataset = load_dataset(opt, alarm_type, explict_sensor)
    if not combined:
        dataset = all_dataset['ALL']
    else:
        dataset = all_dataset[date_name]
    if not dataset:
        return
    opt.n_classes = 2
    opt.input_nc = len(opt.load_sensor_names) if not opt.load_important_sig else 1
    opt.input_length = opt.window_size
    if opt.use_extra:
        opt.extra_length = dataset.get_extro_len()

    model = []
    for i in range(5):
        m = load_model(path=base_path + str(i + 1) + '/' + model_name, opt=opt)
        if m is None:
            # m = load_model(path=base_path + str(i + 1) + '/best.pth', opt=opt)
            return
        model.append(m)
    predict(dataset, model, opt.cuda)
    return


def predict_combined(opt, alarm_type='all', base_path='../checkpoints/deeper_embedded_12s_slice/',
                     model_name="best.pth"):
    opt.cuda = True

    opt.load_sensor_names = ['PLETH', 'RESP', 'ABP']
    predict_testfile(opt, alarm_type=alarm_type, explict_sensor=False,
                     base_path=base_path + 'PRA/', model_name=model_name, date_name='PRA')

    opt.load_sensor_names = ['II']
    predict_testfile(opt, alarm_type=alarm_type, explict_sensor=True,
                     base_path=base_path + 'II/', model_name=model_name, date_name='II')
    opt.load_sensor_names = ['V']
    predict_testfile(opt, alarm_type=alarm_type, explict_sensor=True,
                     base_path=base_path + 'V/', model_name=model_name, date_name='V')

    opt.load_important_sig = True
    predict_testfile(opt, alarm_type=alarm_type, explict_sensor=False,
                     base_path=base_path + 'EEG/', model_name=model_name, date_name='EEG')
    opt.load_important_sig = False

    vote()
    return


def compute_score():
    global all_test_people_poss, all_test_people_labels, all_test_people_results
    tp, tn, fp, fn = 0, 0, 0, 0
    labels = []
    poss = []
    for record, fold in file_dict.items():
        if record not in all_test_people_poss:
            return None
        p, l, t = all_test_people_poss[record], all_test_people_results[record], all_test_people_labels[record]
        if math.isnan(p):
            return None
        if t == l:
            if l == 0:
                tn += 1
            else:
                tp += 1
        else:
            if l == 0:
                fn += 1
            else:
                fp += 1
        labels.append(t)
        poss.append(1 - p)
    acc = (tn + tp) / (tp + tn + fp + fn)
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    score = (tp + tn) / (tp + tn + fp + 5 * fn)
    auc = roc_auc_score(labels, poss)
    return tp, tn, fp, fn, int(tpr * 10000) / 100.0, int(tnr * 10000) / 100.0, int(acc * 10000) / 100.0, int(
        auc * 10000) / 100.0, int(score * 10000) / 100.0


def write_csv_file(filename, res):
    with open(filename, 'a+') as f:
        s = ','.join(res)
        f.write(s + '\n')
    return


def ptest():
    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = False
    opt.model_name = 'deep_embedded'
    opt.window_size = 3750
    opt.use_extra = True
    opt.add_noise_prob = 0

    opt.data_folder = 'H:/ecg_encoder/data/training/'
    opt.train_files = 'H:/ecg_encoder/data/training/test_files_list.txt'
    opt.test_files = 'H:/ecg_encoder/data/training/RECORDS'

    opt.cuda = True
    init_data_dict("H:/ecg_encoder/data/training/")
    init_load_data(opt, combined=True)
    for k in range(0, 401):
        print("model: " + str(k) + ".pth")
        predict_combined(opt, base_path='../checkpoints/deep_embedded_15s/', model_name=str(k) + ".pth")
        # predict_edgcn(opt, base_path='../checkpoints/deep_modified_dgcn_15s/', model_name=str(k) + ".pth")
        # predict_dgcn(opt, base_path='../checkpoints/deep_modified_dgcn_12s_slice_norm_nosing/', model_name=str(k) + ".pth")
        s = compute_score()
        if s is None:
            break
        tp, tn, fp, fn, tpr, tnr, acc, auc, score = s
        tb = pt.PrettyTable()
        res = [str(k), str(tp + tn + fp + fn), str(tp), str(fp), str(tn), str(fn), str(tpr), str(tnr), str(acc),
               str(auc), str(score)]
        tb.field_names = ['Iteration', 'Nums', 'TP', 'FP', 'TN', 'FN', 'TPR', 'TNR', 'ACC', 'AUC', 'Score']
        tb.add_row(res)
        write_csv_file("Combined_3750.csv", res)
        print(tb)
        clear()

    return


def predict_dgcn(opt, base_path="./checkpoints/DGCN/", model_name='best.pth'):
    global test_people_preds

    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    predict_testfile(opt, alarm_type='all', explict_sensor=False,
                     base_path=base_path, model_name=model_name, combined=False)
    vote()
    return


def predict_edgcn(opt, base_path="../checkpoints/deep_embedded_edgcn_12s_slice/", model_name="best.pth"):
    global test_people_preds

    opt.load_sensor_names = ['II', 'V', 'RESP', 'PLETH', 'ABP']
    predict_testfile(opt, model_name=model_name, alarm_type='all', explict_sensor=False,
                     base_path=base_path, combined=False)
    vote()
    return


def clear():
    global all_test_people_poss, all_test_people_results, all_test_people_labels, all_test_people_preds
    all_test_people_results.clear()
    all_test_people_labels.clear()
    all_test_people_preds.clear()
    all_test_people_poss.clear()
    return


if __name__ == "__main__":
    ptest()
    exit(0)
    opt = Opt()
    opt.drop_prob = 0.3
    opt.use_minmax_scale = False
    opt.model_name = 'deep_embedded'
    opt.window_size = 3750
    opt.use_extra = True
    opt.add_noise_prob = 0
    opt.cuda = False

    with open("answers.txt", "a+", encoding="utf-8") as fp:
        for record in sys.argv[1:]:
            output_file = os.path.basename(record)
            results = predict_combined(opt, record)
            fp.write(output_file + ',' + str(results) + "\n")

    sys.exit(0)
