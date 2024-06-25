import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
from tensorboardX import SummaryWriter
from progress.bar import Bar
from tqdm import tqdm
from MixMatchConfig import config
from MixMatch.dataset.cifar10 import get_dataset
from MixMatch.model.wideresnet import WideResNet
from MixMatch.utils.utils import SemiLoss, WeightEMA, AverageMeter, interleave, accuracy


def train():
    '''
    根据config中的参数进行半监督图像分类训练
    :return:
    '''

    '''
    准备cifar10数据集
    '''
    train_labeled_dataset, train_unlabeled_dataset, val_dataset, _ = get_dataset()

    train_labeled_dataloader = DataLoader(
        train_labeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    train_unlabeled_dataloader = DataLoader(
        dataset=train_unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config.batch_size,
        drop_last=False
    )

    '''
    准备网络、损失函数、优化器等
    '''
    # 以WideResNet-28-2作为backbone网络
    model = WideResNet(num_classes=10)
    model.to(config.device)
    # ema_model即为每次训练后参数指数加权平均后的模型，由当前model和历史迭代过程中的model加权的来，本身不参与训练
    ema_model = WideResNet(num_classes=10)  # 参数通过指数平均计算
    ema_model.to(config.device)
    # ema_model本身不参与训练
    for param in ema_model.parameters():
        param.detach_()  # 防止梯度在反向传播时流向这些参数

    # 半监督损失、交叉熵损失以及优化器
    train_loss_fn = SemiLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=config.ema_decay)

    '''
    恢复训练、训练进度记录
    '''
    best_acc = -1.0
    start_epoch = 0
    # 是否恢复训练
    if config.is_resume:
        assert os.path.exists(config.resume_path)
        print('resuming train......')
        checkpoint = torch.load(config.resume_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    writer = SummaryWriter(config.result_path)

    # 开始训练
    model.train()
    for epoch in range(start_epoch, config.total_epoch):
        '''
        =====================Train=====================
        '''
        train_loss = AverageMeter()
        x_loss = AverageMeter()
        u_loss = AverageMeter()

        train_labeled_iter = iter(train_labeled_dataloader)
        train_unlabeled_iter = iter(train_unlabeled_dataloader)

        model.train()
        # 进度条
        bar = Bar(f"[Epoch {epoch}/{config.total_epoch}]Train", max=config.train_iteration)

        # 这里train_iteration，实验要求是20000次迭代，所以epoch数量乘以train_iteration必须是20000
        for batch_idx in range(config.train_iteration):
            # 通过迭代器的方式取出标注数据和未标注数据
            try:
                inputs_x, targets_x = train_labeled_iter.__next__()
            except StopIteration:
                train_labeled_iter = iter(train_labeled_dataloader)
                inputs_x, targets_x = train_labeled_iter.__next__()

            try:
                (inputs_u, inputs_u2), _ = train_unlabeled_iter.__next__()
            except StopIteration:
                train_unlabeled_iter = iter(train_unlabeled_dataloader)
                (inputs_u, inputs_u2), _ = train_unlabeled_iter.__next__()

                # 将标注数据标签转成独热编码
            targets_x = torch.zeros(config.batch_size, 10).scatter_(1, targets_x.view(-1, 1).long(), 1)
            # 使用CPU或GPU
            inputs_x = inputs_x.to(config.device)
            targets_x = targets_x.to(config.device)
            inputs_u = inputs_u.to(config.device)
            inputs_u2 = inputs_u2.to(config.device)

            with torch.no_grad():
                # 预测未标注数据的标签
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                # 通过sharpening函数完成分布的锐化，T->0时该函数的输出趋向于one-hot形式，不过一般设置为0.5左右
                pt = p ** (1 / config.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # 通过mixup完成新数据集的构建，将扩增后的数据拼接
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
            # beta分布中采样一个参数λ
            lam = np.random.beta(config.alpha, config.alpha)
            lam = max(lam, 1 - lam)
            # 生成一个随机排列的整数数列，用于打乱输入(标注+未标注)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = lam * input_a + (1 - lam) * input_b
            mixed_target = lam * target_a + (1 - lam) * target_b
            # 将张量划分为多个batch数据，使得可以输入模型
            mixed_input = list(torch.split(mixed_input, config.batch_size))
            # 将有标签数据和无标签数据交错排列，通过交换元素来增强数据的多样性
            mixed_input = interleave(mixed_input, config.batch_size)

            # 模型前向传播
            # logits = []
            # for input_tensor in mixed_input:
            #     logits.append(model(input_tensor))
            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            # 恢复交错样本，得到标注和未标注的模型结果
            logits = interleave(logits, config.batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            # 计算loss
            Lx, Lu, w = train_loss_fn(
                logits_x, mixed_target[:config.batch_size],
                logits_u, mixed_target[config.batch_size:],
                epoch + batch_idx / config.train_iteration
            )
            loss = Lx + w * Lu

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 注意更新ema_model，这是最终的模型，它是当前参数更新量和之前的参数更新的指数平均
            ema_optimizer.step()

            # 记录loss变化
            train_loss.update(loss.item(), inputs_x.size(0))
            x_loss.update(Lx.item(), inputs_x.size(0))
            u_loss.update(Lu.item(), inputs_x.size(0))

            # 显示训练进度，包括训练loss变化
            bar.suffix = 'Batch: {}/{}; Total Loss: {:.4f}; X Loss: {:.4f}; U Loss: {:.4f}'.format(
                batch_idx + 1,
                config.train_iteration,
                train_loss.avg,
                x_loss.avg,
                u_loss.avg
            )
            bar.next()
        bar.finish()

        '''
        =====================Validate=====================
        '''
        print('validating...')
        ema_model.eval()
        loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        loop.set_description('Validating...')
        val_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in loop:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                targets = targets.long()

                outputs = ema_model(inputs)
                loss = val_loss_fn(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                val_losses.update(loss.item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top5.update(prec5.item(), inputs.shape[0])

        '''
        打印观察、记录训练数据
        '''
        print('train loss: {:.4f}; validate loss: {:.4f}; validate accuracy: {:.4f}\n'.format(
            train_loss.avg, val_losses.avg, top1.avg))

        writer.add_scalar('losses/train_loss', train_loss.avg, epoch)
        writer.add_scalar('losses/val_loss', val_losses.avg, epoch)
        writer.add_scalar('acc/val_acc', top1.avg, epoch)

        is_best = top1.avg > best_acc
        best_acc = max(best_acc, top1.avg)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': top1.avg,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, config.resume_path)
        if is_best:
            torch.save(checkpoint, config.best_path)

    writer.close()
    print('best validate accuracy: ', best_acc)


if __name__ == '__main__':
    train()
