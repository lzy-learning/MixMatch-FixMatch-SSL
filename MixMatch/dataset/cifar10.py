import numpy as np
import math
import torchvision
import torchvision.transforms as transforms
from MixMatchConfig import config
from MixMatch.utils.utils import normalize, RandomFlip, RandomPadandCrop, TransformKTimes, transpose, ToTensor


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    '''
    继承Dataset，重写__getitem__和__len__方法
    '''

    def __init__(self, root, indices=None, train=True, is_labeled=True, transform=None):
        '''
        获取cifar10数据集特定数据（已标注或未标注）
        :param root: 数据集所在位置，默认为./data/
        :param indices: 数据下表，应传入list或None
        :param train: 表明是否是训练集
        :param is_labeled: 表明是否进行标注
        :param transform: 数据增强操作
        '''
        super(CIFAR10Dataset, self).__init__(root, train=train, transform=transform, download=True)
        if indices is not None:
            self.data = self.data[indices]
            # 这里targets属性默认不是ndarray类型，所以不能广播
            self.targets = np.array(self.targets)[indices]
        self.data = transpose(normalize(self.data))
        if not is_labeled:
            self.targets = np.array([-1 for _ in range(len(self.targets))])

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


def get_dataset():
    '''
    获取标注和未标注的训练集、验证集和测试集
    '''
    # 首先需要进行验证集分割，用于检验模型训练有没有过拟合
    cifar10_data = torchvision.datasets.CIFAR10(config.cifar10_path, train=True, download=True)
    labels = np.array(cifar10_data.targets)
    train_labels_indices = []
    train_unlabels_indices = []
    val_labels_indices = []
    for category in range(config.num_classes):  # 十个类别
        indices = np.where(labels == category)[0]
        np.random.shuffle(indices)
        train_labels_indices.extend(indices[:int(config.labeled_num / 10)])
        train_unlabels_indices.extend(indices[int(config.labeled_num / 10):-500])
        # 这里验证集设置为每个类500张图片
        val_labels_indices.extend(indices[-500:])
    np.random.shuffle(train_labels_indices)
    np.random.shuffle(train_unlabels_indices)
    np.random.shuffle(val_labels_indices)
    # 考虑标注图片的数量小于batch_size的情况
    if config.labeled_num < config.batch_size:
        num_expand_x = math.ceil(
            config.batch_size * config.train_iteration / config.labeled_num)
        train_labels_indices = np.hstack([train_labels_indices for _ in range(num_expand_x)])
    np.random.shuffle(train_labels_indices)

    # 训练集需要进行数据增强，进行k=2次
    train_transform = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        ToTensor(),
    ])

    # 验证集只需要转化为张量
    val_transform = transforms.Compose([
        ToTensor()
    ])

    # 标注训练集
    train_labeled_dataset = CIFAR10Dataset(
        root=config.cifar10_path, indices=train_labels_indices,
        train=True, is_labeled=True,
        transform=train_transform
    )
    # 未标注训练集
    train_unlabeled_dataset = CIFAR10Dataset(
        root=config.cifar10_path, indices=train_unlabels_indices,
        train=True, is_labeled=False,
        transform=TransformKTimes(train_transform, config.K)
    )
    # 验证集
    val_dataset = CIFAR10Dataset(
        root=config.cifar10_path, indices=val_labels_indices,
        train=True, is_labeled=True,
        transform=val_transform
    )
    # 测试集
    test_dataset = CIFAR10Dataset(
        root=config.cifar10_path, train=False,
        transform=val_transform
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
