# Automatic Network Pruning via Information Bottleneck Minimization.


### Model Pruning

##### 1. VGG-16
pruning ratio (FLOPs): 60%
```shell
python main.py \
--model vgg16\
--dataset cifar10\
--target 126000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 5e-5
```
##### 2. ResNet56
pruning ratio (FLOPs): 55%
```shell
python main.py \
--model resnet56\
--dataset cifar10\
--target 57000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 8e-4
```
##### 3. ResNet110 
pruning ratio (FLOPs): 63%
```shell
python main.py \
--model resnet110\
--dataset cifar10\
--target 94000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 8e-9
```
##### 4. GoogLeNet
pruning ratio (FLOPs): 63%
```shell
python main.py \
--model googlenet\
--dataset cifar10\
--target 568000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 4e-8
```
##### 5. ResNet50
pruning ratio (FLOPs): 62%

```shell
python main.py \
--model resnet50\
--dataset imagenet\
--target 1550000000 \
--ckpt [pre-trained model dir] \
--data_path [dataset path]\
--omega 1\
--tolerance 0.01\
--alpha 7e-5
```
### Model Training
##### 1. VGG-16
```shell
python train.py \
--model vgg16\
--dataset cifar10\
--lr 0.1\
--batch_size 128 \
--ckpt_path [pruned model dir]\
--data_path [dataset path]
```
##### 2. ResNet-50
```shell
python train.py \
--model resnet50\
--dataset imagenet\
--lr 0.025\
--batch_size 128 \
--ckpt_path [pruned model dir]\
--data_path [dataset path]
```
## Pre-trained Models 

Additionally, we provide the pre-trained models used in our experiments. 


### CIFAR-10:
 [Vgg-16](https://drive.google.com/file/d/1g9Yz9mABWYXXRWpyN5foA5NQc7JvunjY/view?usp=sharing) 
| [ResNet56](https://drive.google.com/file/d/1vJ5lXoW8RJF6_ZA_pdCrVIem2RbkYo5h/view?usp=share_link) 
| [ResNet110](https://drive.google.com/file/d/1hwo4JZGOn3zKoGSTefVQLa5vnNtsdIdn/view?usp=share_link)  
| [GoogLeNet](https://drive.google.com/file/d/1kg8ndpwGaMorrqRVPAic20qwAM21d01-/view?usp=share_link) 
### CIFAR-100:
 [Vgg-16](https://drive.google.com/file/d/1DZns2H-KrVdpndPLO6s0vjo0feoRv8w6/view?usp=share_link) 
| [ResNet56](https://drive.google.com/file/d/18EbAD6-E-t1Dk-x2tKkoWfJ86I3iXgFa/view?usp=share_link)

### ImageNet:
 [ResNet50](https://drive.google.com/file/d/1pWYDy9nDDWpflsOiS-b4rxFNMZ5X-Nds/view?usp=share_link)

## Acknowledgments

Our implementation partially reuses [Lasso's code](https://github.com/lippman1125/channel_pruning_lasso) | [HRank's code](https://github.com/lmbxmu/HRank) | [ITPruner's code](https://github.com/MAC-AutoML/ITPruner).