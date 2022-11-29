import os
import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import numpy as np
from dataset import cifar10,cifar100
from tensorboardX import SummaryWriter
from models.resnet_imagenet import resnet50, resnet50_X
from models.vgg_cifar import vgg16,vgg16_X
from models.resnet_cifar import resnet20,resnet20_X,resnet44,resnet44_X,resnet110,resnet110_X,resnet56,resnet56_X
from models.googlenet import googlenet,googlenet_X
from utils.utils import accuracy, AverageMeter, progress_bar
from thop import profile
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='finetune')
    parser.add_argument('--model', default=None, type=str, help='name of the model to train')
    parser.add_argument('--dataset', default=None, type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=8, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='cos', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=135, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--cfg', default=None,type=str, help='channel number of each layer')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--mixup', default=False,action='store_true', help='use mixup data augmentation')
    parser.add_argument('--prune_layer', nargs="+", default=None, help='layer to prune')
    parser.add_argument('--data_path',type=str,default=None,help='The dictionary where the input is stored. default:')

    return parser.parse_args()


def get_model():
    print('=> Building model..')
    if args.model == 'vgg16':
        net = vgg16()
    elif args.model == 'vgg16_X':
        net = vgg16_X(eval(args.cfg))
    elif args.model == 'resnet20':
        net = resnet20()
    elif args.model == 'resnet20_X':
        net = resnet20_X(eval(args.cfg))
    elif args.model == 'resnet56_X':
        net = resnet56_X(eval(args.cfg))
    elif args.model == 'resnet56':
        net = resnet56()
    elif args.model == 'resnet50':
        net = resnet50()
    elif args.model == 'resnet50_X':
        net = resnet50_X(eval(args.cfg))
    elif args.model == 'resnet110':
        net = resnet110()
    elif args.model == 'resnet110_X':
        net = resnet110_X(eval(args.cfg))
    elif args.model == 'resnet44':
        net = resnet44()
    elif args.model == 'resnet44_X':
        net = resnet44_X(eval(args.cfg))
    elif args.model == 'resnet50':
        net = resnet50()
    elif args.model == 'resnet50_X':
        net = resnet50_X(eval(args.cfg))
    elif args.model == 'googlenet':
        net = googlenet()
    elif args.model == 'googlenet_X':
        net = googlenet_X(eval(args.cfg))
    else:
        raise NotImplementedError
    return net.cuda() if use_cuda else net

def mixup_data(x, y, alpha=0.086, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        if not args.mixup:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        else:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda=use_cuda)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        if not args.mixup:
            top1.update(prec1.item(), inputs.size(0))
        else:
            _, predicted = torch.max(outputs.data, 1)
            prec1 = (lam * predicted.eq(targets_a.data).cpu().sum().float()
                     + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            prec1 = prec1 * 100 / inputs.size(0)
            top1.update(prec1, inputs.size(0))

        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))
        # disp_mask(net, args.prune_layer)
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)


def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.n_epoch))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

if __name__ == '__main__':
    args = parse_args()
    args.gpus=[0,1,2,3]
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    device = 'cuda' if use_cuda else 'cpu'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    print('=> Preparing data..')
    if args.dataset == "cifar10":
        loader = cifar10.Data(args)
        train_loader=loader.trainLoader
        test_loader=loader.testLoader
    elif args.dataset == "cifar100":
        loader = cifar100.Data(args)
        train_loader=loader.trainLoader
        test_loader=loader.testLoader
    elif args.dataset == "imagenet":
        traindir = os.path.join(args.data_path, 'train')
        valdir   = os.path.join(args.data_path, 'val')
        scale_size =224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(scale_size),
                        transforms.ToTensor(),
                        normalize,
                    ]))

        train_loader = DataLoader(trainset,batch_size=64,shuffle=True,num_workers=8,pin_memory=True)

        testset = datasets.ImageFolder(
                    valdir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Resize(scale_size),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        test_loader = DataLoader(
                    testset,
                    batch_size=64,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True)

    net = get_model()  # for measure
    IMAGE_SIZE = 224 if args.dataset == 'imagenet' else 32
    dummy = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)
    n_flops, n_params = profile(net, (dummy, ), verbose=False)
    print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M'.format(n_params / 1e6, n_flops / 1e6))
    del net

    net = get_model()
    if args.ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(args.ckpt_path)
        checkpoint = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        net.load_state_dict(checkpoint)

    if use_cuda and args.n_gpu > 1:
        net = torch.nn.DataParallel(net, list(range(args.n_gpu)))
    elif use_cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    if args.eval:  # just run eval
        print('=> Start evaluation...')
        test(0, test_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(args.model, args.dataset))
        log_dir = get_output_folder('./logs', '{}_{}_finetune'.format(args.model, args.dataset))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader)
            test(epoch, test_loader)

        writer.close()
        print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M, best top-1 acc: {}%'.format(n_params / 1e6,
                                                                                         n_flops / 1e6, best_acc))
