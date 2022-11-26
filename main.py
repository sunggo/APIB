import os
import argparse
from models.googlenet import googlenet_X
from models.resnet_cifar import resnet110

from prune import HSICLassoPruner
from config import HSICLassoPruneConfig
import logging
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from pruning_policy import pruning_policy
from models.vgg_cifar import vgg16
from models.resnet_cifar import resnet20,resnet110,resnet44,resnet56
from models.resnet_imagenet import resnet50
from models.googlenet import googlenet
from dataset import cifar10,cifar100
from torch.utils.data import DataLoader
from thop import profile
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('checkpoints/', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
os.environ['CUDA_VISIBLE_DEVICES']='1'
def get_config():
    print('=> Building model..')

    if args.model == 'vgg16':
        net, ratios, policy = vgg16(), None, pruning_policy
    elif args.model == 'resnet20':
        net, ratios, policy = resnet20(), None, pruning_policy
    elif args.model == 'resnet56':
        net, ratios, policy = resnet56(), None, pruning_policy
    elif args.model == 'resnet44':
        net, ratios, policy = resnet44(), None, pruning_policy
    elif args.model == 'resnet110':
        net, ratios, policy = resnet110(), None, pruning_policy
    elif args.model == 'googlenet':
        net, ratios, policy = googlenet(),None, pruning_policy
    elif args.model == 'resnet50':
        net, ratios, policy = resnet50(), None, pruning_policy
    else:
        print("Not support model {}".format(args.model))
        raise NotImplementedError
    return net, ratios, policy

def check_args(args):
    print("=> Checking Parameter")
    ret = 0
    if not os.path.exists(args.calib_dir):
        print("calib dir {} not exists".format(args.calib_dir))
        ret = -1
    if not os.path.exists(args.valid_dir):
        print("valid dir {} not exists".format(args.valid_dir))
        ret = -1
    if not os.path.exists(args.ckpt):
        print("checkpoint {} not exists".format(args.ckpt))
        ret = -1
    return ret

parser = argparse.ArgumentParser(description='Channel pruning')
parser.add_argument('--model', default='vgg16', type=str, help='name of the model to train')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--calib_batch', default=0, type=int, help='how many batches used to calib')
parser.add_argument('--n_worker', default=8, type=int, help='number of data loader worker')
parser.add_argument('--alpha', default=1e-6, type=float, help='global penalty coifficient')
parser.add_argument('--tolerance', default=0.01, type=float, help='Tu and Tl')
parser.add_argument('--omega', default=1, type=int, help='threshold')
parser.add_argument('--target', type=int, help='global penalty coifficient')
parser.add_argument('--seed', default=None, type=int, help='random seed to set')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint path to resume from')
parser.add_argument('--calib_dir', default='checkpoints/calib', type=str, help='calib dataset path')
parser.add_argument('--valid_dir', default='checkpoints/valid', type=str, help='valid dataset path')
parser.add_argument('--pruner', default='HSIC_lasso', type=str)
parser.add_argument('--fmap', default=None, type=str, help='feature map file')
parser.add_argument('--fmap_save', action='store_true', help='save feature map')
parser.add_argument('--fmap_save_path', default='./', type=str, help='feature map save path')
parser.add_argument('--data_path',type=str,default='datasets/cifar',help='The dictionary where the input is stored. default:')
parser.add_argument('--dataset', default='cifar10', type=str, help='name of the dataset to train')

args = parser.parse_args()
args.gpus=[0,1,2,3]
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
print(args)

if check_args(args) < 0:
    print("paramters check fail")
    exit(0)
if args.dataset == "cifar10":
    loader = cifar10.Data(args)
    train_loader=loader.trainLoader
    val_loader=loader.testLoader
elif args.dataset == "cifar100":
    loader = cifar100.Data(args)
    train_loader=loader.trainLoader
    val_loader=loader.testLoader
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
    val_loader = DataLoader(
                testset,
                batch_size=64,
                shuffle=False,
                num_workers=8,
                pin_memory=True)
net, sparsity_ratios, pruning_policy = get_config()

HSIClassopruner_config = HSICLassoPruneConfig(args.model,
                                      net,
                                      args.ckpt,
                                      train_dataloader=train_loader,
                                      pruner=args.pruner,
                                      val_dataloader=val_loader,
                                      criterion=nn.CrossEntropyLoss().cuda(),
                                      policy=pruning_policy,
                                      fmap_path=args.fmap)
HSIClassopruner_config.calib_batch = args.calib_batch
HSIClassopruner_config.fmap_save = args.fmap_save
HSIClassopruner_config.fmap_save_path = args.fmap_save_path

HSIClassopruner = HSICLassoPruner(HSIClassopruner_config)
HSIClassopruner.metric()
HSIClassopruner.auto_prune(args.model,alpha=args.alpha,target_params=args.target,tolerance=args.tolerance,threshold=args.omega)
HSICLassoPruner.save_pruned_model('./')




