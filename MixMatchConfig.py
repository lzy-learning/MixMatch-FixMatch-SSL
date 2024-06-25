import os
import sys
import torch


class Config:
    def __init__(self):
        self.cifar10_path = 'data'
        self.num_classes = 10

        self.total_epoch = 20
        self.start_epoch = 0
        self.train_iteration = 1000
        self.total_steps = self.total_epoch * self.train_iteration

        self.batch_size = 64
        self.lr = 0.002

        # 恢复训练
        self.is_resume = False
        self.resume_path = 'mixmatch_result/checkpoint.pth.tar'
        self.best_path = 'mixmatch_result/best.pth.tar'

        self.labeled_num = 4000

        self.result_path = 'mixmatch_result/logdir'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.alpha = 0.75
        self.lambda_u = 75
        self.T = 0.5
        self.ema_decay = 0.999

        self.K = 2

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


config = Config()
