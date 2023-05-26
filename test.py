import argparse
import time
import json
import os
from tqdm import tqdm
from models import *
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from ResNet34 import ResNet34

# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.model='ResNet34'
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        kwargs = {
            'num_workers': args.num_workers,
            'pin_memory': True,
        } if args.is_cuda else {}


        # 载入数据
        test_dataset = datasets.ImageFolder(
            args.test_dir,
        transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,  # 是DataLoader类的一个参数，指定了要使用的数据集
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

        # 创建ResNet34模型实例
        model = ResNet34(num_classes=args.num_classes)

        self.model = model.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

        # warm up 学习率调整部分
        self.per_epoch_size = len(test_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0



    def test(self,epoch):
        self.model.eval()
        test_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.test_loader)
            for data, target in bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_function(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        test_loss /= len(self.test_loader.dataset)


        print('test >> Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            num_correct,
            len(self.test_loader.dataset),
            100. * num_correct / len(self.test_loader.dataset)
        ))
        return test_loss
        # 返回重要信息，用于生成模型保存命名
        # return 100. * num_correct / len(self.test_loader.dataset),


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)  # 用于创建一个 Worker 类的实例。
                                # worker是一个Worker类的实例，Worker(args=args)是一个类构造函数，它用于创建Worker类的实例。

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.test(epoch)
        test_loss = worker.test(epoch)
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, test_loss)
            torch.save(worker.model, save_dir)
