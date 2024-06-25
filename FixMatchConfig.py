import os
import sys
import torch
import math


class Config:
    def __init__(self):
        self.cifar10_path = 'data'

        self.start_epoch = 0
        self.total_epoch = 20
        self.batch_size = 64
        self.lr = 0.002
        self.weight_decay = 5e-4
        self.labeled_num = 4000

        # 恢复训练
        self.is_resume = False
        self.resume_path = f'fixmatch_result/checkpoint_{self.labeled_num}.pth.tar'
        self.best_path = f'fixmatch_result/best_{self.labeled_num}.pth.tar'

        self.result_path = 'fixmatch_result/logdir'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # 指数加权平均算法中的权重系数
        self.ema_decay = 0.999
        self.lambda_u = 1
        self.T = 1
        self.threshold = 0.95
        # 未标注数据的batch大小的权重系数
        self.mu = 7
        # 每隔多少个epoch进行一次测试
        self.train_iteration = 1000
        # 总的迭代次数为20000
        self.total_steps = self.train_iteration * self.total_epoch

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


config = Config()
