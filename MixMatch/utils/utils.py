import numpy as np
import torch
import torch.nn.functional as F
from MixMatchConfig import config


def train_val_split(cifar10_labels):
    '''
    划分训练集和验证集，验证集固定为500张图片，一共10个类别，每个类别的标注数在config中指定
    :param cifar10_labels: CIFAR10数据集的标签
    :return:
    '''
    cifar10_labels = np.array(cifar10_labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(cifar10_labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:config.labeled_num])
        train_unlabeled_idxs.extend(idxs[config.labeled_num:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class SemiLoss(object):
    '''
    MixMatch半监督学习的损失函数
    '''

    def linear_rampup(self, current, rampup_length=config.total_epoch):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, config.lambda_u * self.linear_rampup(epoch)


class WeightEMA(object):
    '''
    更新ema_model参数
    '''

    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * config.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                param.mul_(1 - self.wd)


class AverageMeter(object):
    '''
    统计平均值，可以是时间或者损失函数值的平均值
    '''

    def __init__(self):
        self.reset()
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    '''
    计算precision@k
    :param output:
    :param target:
    :param topk:
    :return:
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interleave_offsets(batch, num):
    '''
    将一个批次的数据分成交错的多个部分
    :param batch:
    :param num:
    :return:
    '''
    # groups表示将batch拆分成nu+1个部分后的每部分的大小
    groups = [batch // (num + 1)] * (num + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    '''
    将数据交错排列
    :param xy:
    :param batch:
    :return:
    '''
    num = len(xy) - 1
    offsets = interleave_offsets(batch, num)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(num + 1)] for v in xy]
    for i in range(1, num + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    # 将分割后的数据重新拼接回去
    return [torch.cat(v, dim=0) for v in xy]


def normalize(x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class ToTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class TransformKTimes:
    def __init__(self, transform, k):
        self.transform = transform
        self.k = k

    def __call__(self, img):
        res = []
        for _ in range(self.k):
            res.append(self.transform(img))
        return tuple(res)
