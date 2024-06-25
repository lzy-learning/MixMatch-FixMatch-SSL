import os

import torch
import torch.nn as nn
from MixMatchConfig import config
from MixMatch.dataset.cifar10 import get_dataset
from MixMatch.utils.utils import AverageMeter, accuracy
from MixMatch.model.wideresnet import WideResNet
from tqdm import tqdm
from torch.utils.data import DataLoader


def test():
    ''''''

    '''
    准备数据集
    '''
    _, _, _, test_dataset = get_dataset()
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )

    '''
    加载模型参数
    '''
    model = WideResNet(num_classes=10)
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
    loop.set_description('MixMatch Testing...')
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
