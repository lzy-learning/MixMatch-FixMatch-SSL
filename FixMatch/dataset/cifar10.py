import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from FixMatch.utils.utils import RandAugmentMC
from FixMatchConfig import config


def get_cifar10():
    '''
    获取标注训练集、未标注训练集、验证集、测试集
    :return:
    '''

    # 训练集要进行数据增强操作，分为strong augment和weak augment
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=4,
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])
    base_dataset = datasets.CIFAR10(config.cifar10_path, train=True, download=True)

    # 划分标注训练集、未标注训练集和验证集，获取它们的索引
    train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs = train_val_split(
        base_dataset.targets, int(config.labeled_num / 10))

    # 创建对应Dataset
    train_labeled_dataset = CIFAR10Dataset(config.cifar10_path, train_labeled_idxs, train=True,
                                           transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10Dataset(
        config.cifar10_path, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)))

    val_dataset = CIFAR10Dataset(
        config.cifar10_path, train_unlabeled_idxs, train=True,
        transform=transform_val
    )
    test_dataset = datasets.CIFAR10(
        config.cifar10_path, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, ):
        super().__init__(root, train=train, transform=transform, download=True)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def train_val_split(labels, n_labeled_per_class):
    '''
    划分标注训练、未标注训练集以及验证集，和mixmatch中的一样
    :param labels:
    :param n_labeled_per_class:
    :return:
    '''
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    # 考虑标注图片的数量小于batch_size的情况
    if config.labeled_num < config.batch_size:
        num_expand_x = math.ceil(config.batch_size * config.eval_step / config.labeled_num)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(num_expand_x)])
    np.random.shuffle(train_labeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
