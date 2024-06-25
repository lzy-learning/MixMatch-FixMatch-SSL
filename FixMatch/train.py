import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

from FixMatch.model.wideresnet import WideResNet
from FixMatch.dataset.cifar10 import get_cifar10
from FixMatchConfig import config
from FixMatch.utils.utils import ModelEMA, AverageMeter, interleave, de_interleave, accuracy


def train():
    '''
    数据集处理
    '''
    train_labeled_dataset, train_unlabeled_dataset, val_dataset, _ = get_cifar10()
    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=RandomSampler(train_labeled_dataset),
        batch_size=config.batch_size,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset,
        sampler=RandomSampler(train_unlabeled_dataset),
        batch_size=config.batch_size * config.mu,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config.batch_size,
        drop_last=False
    )

    best_acc = -1.0

    '''
    模型、损失函数、优化器
    '''
    model = WideResNet(depth=28, widen_factor=2, dropRate=0, num_classes=10)
    model.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # 仍使用指数加权平均的方式获取最终参数
    ema_model = ModelEMA(model, config.ema_decay)

    '''
    恢复训练
    '''
    if config.is_resume:
        assert os.path.exists(config.resume_path)
        print('resuming train......')
        checkpoint = torch.load(config.resume_path)
        best_acc = checkpoint['best_acc']
        config.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    '''
    开始训练
    '''
    writer = SummaryWriter(config.result_path)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    model.train()
    for epoch in range(config.start_epoch, config.total_epoch):
        '''
        训练
        '''
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        # 进度条
        bar = Bar(f"[Epoch {epoch}/{config.total_epoch}] Train", max=config.train_iteration)

        for batch_idx in range(config.train_iteration):
            # 通过迭代器的方式取出标注数据和未标注数据
            try:
                inputs_x, targets_x = labeled_iter.__next__()
            except StopIteration:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.__next__()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.__next__()
            except StopIteration:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.__next__()

            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * config.mu + 1)
            inputs = inputs.to(config.device)
            targets_x = targets_x.to(config.device).long()

            logits = model(inputs)
            logits = de_interleave(logits, 2 * config.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / config.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(config.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = Lx + config.lambda_u * Lu

            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()

            ema_model.update(model)
            model.zero_grad()

            # 显示训练进度，包括训练loss变化
            bar.suffix = 'Batch: {}/{}; Total Loss: {:.4f}; X Loss: {:.4f}; U Loss: {:.4f}'.format(
                batch_idx + 1,
                config.train_iteration,
                losses.avg,
                losses_x.avg,
                losses_u.avg
            )
            bar.next()
        bar.finish()

        '''
        测试
        '''
        test_model = ema_model.ema
        test_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        test_model.eval()
        bar = Bar(f"[Epoch {epoch}/{config.total_epoch}] Test", max=len(val_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                outputs = test_model(inputs)
                targets = targets.long()
                loss = F.cross_entropy(outputs, targets)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                test_losses.update(loss.item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top5.update(prec5.item(), inputs.shape[0])

                bar.suffix = 'Batch: {}/{}; Test Loss: {:.4f}; top1: {:.4f}; top5: {:.4f}'.format(
                    batch_idx + 1,
                    len(val_loader),
                    test_losses.avg,
                    top1.avg,
                    top5.avg
                )
                bar.next()
            bar.finish()

        '''
        保存恢复点
        '''
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.ema.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, config.resume_path)

        writer.add_scalar('train/train_loss', losses.avg, epoch)
        writer.add_scalar('train/train_loss_x', losses_x.avg, epoch)
        writer.add_scalar('train/train_loss_u', losses_u.avg, epoch)
        writer.add_scalar('test/test_loss', test_losses.avg, epoch)
        writer.add_scalar('test/test_acc', top1.avg, epoch)

        is_best = top1.avg > best_acc
        best_acc = max(top1.avg, best_acc)
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.ema.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(config.result_path, 'best.pth.tar'))
    writer.close()
    print("best accuracy in validate dataset: {:.4f}".format(best_acc))


if __name__ == '__main__':
    train()
