import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from FixMatch.model.wideresnet import WideResNet
from FixMatch.dataset.cifar10 import get_cifar10
from FixMatchConfig import config
from FixMatch.utils.utils import AverageMeter, accuracy

def test():
    ''''''
    '''
    准备数据集
    '''
    _, _, _, test_dataset = get_cifar10()

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )

    '''
    加载模型
    '''
    model = WideResNet(depth=28, widen_factor=2, dropRate=0, num_classes=10)
    model.to(config.device)
    if not os.path.exists(config.best_path):
        assert "Train first!"
    checkpoint = torch.load(config.best_path)
    # 注意是加载ema模型参数，不是state_dict
    model.load_state_dict(checkpoint['ema_state_dict'])

    '''
    测试
    '''
    model.eval()
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    loop.set_description('FixMatch Testing...')
    test_losses = AverageMeter()
    acc = AverageMeter()
    test_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in loop:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            targets = targets.long()

            outputs = model(inputs)
            loss = test_loss_fn(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            test_losses.update(loss.item(), inputs.shape[0])
            acc.update(prec1.item(), inputs.shape[0])

    print('test loss: {:.4f}; test accuracy: {:.4f}\n'.format(test_losses.avg, acc.avg))
