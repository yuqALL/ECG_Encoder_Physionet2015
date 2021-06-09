import numpy as np
import torch
from torch import nn
from torch.nn import init


# def initialize_weights(net_l):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv1d):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias.data, 0.0)


def initialize_weights(net_l):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_params(model, batch_norm=True):
    param_dict = dict(list(model.named_parameters()))
    conv_weight = param_dict["conv.weight"]
    init.xavier_uniform_(conv_weight, gain=1)
    if not batch_norm:
        conv_bias = param_dict["conv.bias"]
        init.constant_(conv_bias, 0)
    else:
        bnorm_weight = param_dict["bnorm.weight"]
        bnorm_bias = param_dict["bnorm.bias"]
        init.constant_(bnorm_weight, 1)
        init.constant_(bnorm_bias, 0)


def to_dense_prediction_model(model, axis=(2, 3)):
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.

    Parameters
    ----------
    model
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).
    
    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.

    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def wrap(y, dtype='float'):
    y_wrap = Variable(torch.from_numpy(y))
    if dtype == 'float':
        y_wrap = y_wrap.float()
    elif dtype == 'long':
        y_wrap = y_wrap.long()

    if torch.cuda.is_available():
        y_wrap = y_wrap.cuda()
        return y_wrap


def unwrap(y_wrap):
    if y_wrap.is_cuda:
        y = y_wrap.cpu().data.numpy()
    else:
        y = y_wrap.data.numpy()
    return y


def wrap_X(X):
    X_wrap = copy.deepcopy(X)
    for jet in X_wrap:
        jet["content"] = wrap(jet["content"])
    return X_wrap


def unwrap_X(X_wrap):
    X_new = []
    for jet in X_wrap:
        jet["content"] = unwrap(jet["content"])
        X_new.append(jet)
    return X_new


##Batchization of the jets using LOUPPE'S code
def batch(jets):
    jet_children = []  # [n_nodes, 2]=> jet_children[nodeid, 0], jet_children[nodeid, 1]
    offset = 0
    for j, jet in enumerate(jets):
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset
        jet_children.append(tree)
        offset += len(tree)

    jet_children = np.vstack(jet_children)
    jet_contents = torch.cat([jet["content"] for jet in jets], 0)  # [n_nodes, n_features]
    n_nodes = offset

    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32)
    level_children[:, [0, 2]] -= 1

    inners = []  # Inner nodes at level i
    outers = []  # Outer nodes at level i
    offset = 0

    for jet in jets:
        queue = [(jet["root_id"] + offset, -1, True, 0)]

        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0)

            if len(inners) < depth + 1:
                inners.append([])
            if len(outers) < depth + 1:
                outers.append([])

            if jet_children[node, 0] != -1:  # Inner node
                inners[depth].append(node)
                position = len(inners[depth]) - 1
                is_leaf = False

                queue.append((jet_children[node, 0], node, True, depth + 1))
                queue.append((jet_children[node, 1], node, False, depth + 1))

            else:  # Outer node
                outers[depth].append(node)
                position = len(outers[depth]) - 1
                is_leaf = True

            if parent >= 0:  # Register node at its parent
                if is_left:
                    level_children[parent, 0] = position
                    level_children[parent, 1] = is_leaf
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"])

    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []

    prev_inner = np.array([], dtype=int)

    for inner, outer in zip(inners, outers):
        n_inners.append(len(inner))
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        level = np.concatenate((inner, outer))
        level = torch.from_numpy(level)
        if torch.cuda.is_available(): level = level.cuda()
        levels.append(level)

        left = prev_inner[level_children[prev_inner, 1] == 1]
        level_children[left, 0] += len(inner)
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)

        contents.append(jet_contents[levels[-1]])

        prev_inner = inner

    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 2]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 1] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]

    level_children = torch.from_numpy(level_children).long()
    n_inners = torch.from_numpy(np.array(n_inners)).long()
    if torch.cuda.is_available():
        level_children = level_children.cuda()
        n_inners = n_inners.cuda()

    return (levels, level_children[:, [0, 2]], n_inners, contents)
