from models.googlenet import googlenet,googlenet_X
from models.resnet_cifar import resnet44, resnet44_X, resnet56, resnet56_X,resnet110,resnet110_X,resnet20,resnet20_X
from models.resnet_imagenet import resnet50, resnet50_X
from models.vgg_cifar import vgg16_X,vgg16
model={
    "vgg16":vgg16,
    "resnet20":resnet20,
    "resnet44":resnet44,
    "resnet56":resnet56,
    "resnet110":resnet110,
    "resnet50":resnet50,
    "googlenet":googlenet,
    "vgg16_X":vgg16_X,
    "resnet20_X":resnet20_X,
    "resnet44_X":resnet44_X,
    "resnet56_X":resnet56_X,
    "resnet110_X":resnet110_X,
    "resnet50_X":resnet50_X,
    "googlenet_X":googlenet_X
}
