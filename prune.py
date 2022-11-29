from mimetypes import init
import os
from xml.dom import INVALID_MODIFICATION_ERR
import torch
import numpy as np
import torch.nn as nn
from models.googlenet import googlenet,googlenet_X
from models.resnet_cifar import resnet44, resnet44_X, resnet56, resnet56_X,resnet110,resnet110_X,resnet20,resnet20_X
from models.resnet_imagenet import resnet50, resnet50_X
from models.vgg_cifar import vgg16_X,vgg16
from pruners.factory import get_pruner
# from reconstruct import prune_next_layer
from abc import ABCMeta, abstractclassmethod
from copy import deepcopy
from utils.utils import AverageMeter, accuracy, progress_bar
from thop import profile
from models.model import model
class Pruner(metaclass=ABCMeta):

    @abstractclassmethod
    def prune(self, ratios):
        pass


class HSICLassoPruner(Pruner):
    def __init__(self, config):
        super(HSICLassoPruner, self).__init__()
        self.config = config
        self.device = config.device
        self.model = config.model.to(self.device)
        self.ckpt = config.ckpt
        self.train_dataloader = config.train_dataloader
        self.val_dataloader = config.val_dataloader
        self.n_points_per_layer = config.n_points_per_layer
        self.prunable_layer_types = config.prunable_layer_types
        self.calib_batch = config.calib_batch
        self.criterion = config.criterion
        self.policy = config.policy
        self.pruner = get_pruner(self.config.pruner)
        self._load_checkpoint()
        self._build_index()
        self.pruning_info = list()
        self.stayed_indices=[]
        self.pruning_strategy={
            "vgg16":self.vgg_cifar_auto_prune_layer,
            "resnet56":self.resnet_cifar_auto_prune_layer,
            "resnet110":self.resnet_cifar_auto_prune_layer,
            "resnet50":self.resnet_imagenet_auto_prune_layer,
            "googlenet":self.googlenet_auto_prune_layer
        }
        if self.config.fmap_path is not None:
            self._load_layer_info(self.config.fmap_path)
        else:
            self._extract_layer_info()

    def set_method(self):
        pass

    def _load_checkpoint(self):
        assert os.path.exists(self.ckpt)
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items() if not (k.endswith('total_ops') or k.endswith('total_params')) }
        self.model.load_state_dict(checkpoint)
        self.pruned_model = deepcopy(self.model)

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.prunable_names=[]
        self.layer_type_dict = {}
        flag=0
        # build index and the min strategy dict
        for i,(n,m) in enumerate(self.model.named_modules()):
            if type(m) == nn.AvgPool2d:    #vgg
                self.prunable_idx.append(i)
                self.prunable_ops.append(m)
                self.prunable_names.append(n)
                self.layer_type_dict[i] = type(m)
            if type(m) in self.prunable_layer_types:
                # we do not prune depthwise conv
                #if type(m) == nn.Conv2d or type(m) == nn.Linear:
                    if flag == 1:
                        break
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.prunable_names.append(n)
                    self.layer_type_dict[i] = type(m)
                    if type(m) == nn.Linear:
                        flag=1

        for i in range(len(self.prunable_idx)):
            print('=> Prunable layer idx: {} op type: {} name: {}'.format(self.prunable_idx[i], self.prunable_ops[i],self.prunable_names[i]))

    def _load_layer_info(self, path):
        print("=> load layer info")
        from utils.fmap_load import fmap_load
        self.layer_info_dict = fmap_load(path)

    def _extract_layer_info(self):
        m_list = list(self.model.modules())

        self.layer_info_dict = dict()
        for idx in self.prunable_idx:
            self.layer_info_dict[idx] = dict()

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_dataloader):  # use image from train set
                if i_b > self.calib_batch:
                    break
                input_var = torch.autograd.Variable(input).to(self.device)

                # inference and collect stats
                _ = self.model(input_var)

                # first conv exclude, because we do not prune input channel
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()
                    # conv
                    if len(f_in_np.shape) == 4:
                        if type(m_list[idx])==nn.AvgPool2d:
                                f_in2save=f_in_np.copy()
                                f_out2save=f_out_np.copy()
                        else:
                            b, i_c, i_h, i_w = f_in_np.shape
                           
                            f_in2save= f_in_np.reshape(b,i_c,-1)
                            f_out2save=f_out_np.reshape(b,-1)
                            
                    # fc
                    else:  # first linear 
                        
                        assert len(f_in_np.shape) == 2
                        #mobilenetv1不需要
                        pre_idx=idx-1
                        while pre_idx>=0:
                            if(type(m_list[pre_idx])==nn.AvgPool2d):
                                break;
                            pre_idx-=1
                        if pre_idx>=0:
                            f_in_np=m_list[pre_idx].output_feat.data.cpu().numpy()
                            b, i_c, i_h, i_w = f_in_np.shape
                            self.prunable_idx.remove(pre_idx)
                        else:
                            b, i_c = f_in_np.shape
                        f_in2save= f_in_np.reshape(b,i_c,-1)
                        f_out2save = f_out_np.copy()
                        
                        
                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

        for idx in self.prunable_idx:
            print('Layer NO.{} {}'.format(idx, m_list[idx].__class__.__name__))
            print('\tinput_feat shape : {}'.format(self.layer_info_dict[idx]['input_feat'].shape))
            print('\toutput_feat shape : {}'.format(self.layer_info_dict[idx]['output_feat'].shape))

        if self.config.fmap_save:
            import pickle
            with open(os.path.join(self.config.fmap_save_path, "fmap_5000.pkl"), 'wb') as f:
                pickle.dump(self.layer_info_dict, f, pickle.HIGHEST_PROTOCOL)

    def _record_pruning_layer(self, idx, op, orig_chn, remain_chn):
        pruning_unit = dict()
        pruning_unit['layer idx'] = idx
        pruning_unit['orig_chn'] = orig_chn
        pruning_unit['remain_chn'] = remain_chn
        op_type = 'Unkown'
        if type(op) == torch.nn.Conv2d:
            op_type = 'Conv2d'
        elif type(op) == torch.nn.Linear:
            op_type = 'Linear'
        pruning_unit['type'] = op_type
        self.pruning_info.append(pruning_unit)

    def _prune_prev_layer(self, layer_ind, weights, filter_inds):
        if self.policy is not None:
            self.policy(self.pruned_model, layer_ind, weights, filter_inds, self.device)
    def prune_next_layer(self,X, Y, op, keep_inds, debug=False):
 
        W = op.weight.data.cpu().numpy()
        # conv
        if len(W.shape) == 4:
            if op.groups>1:
                rec_weight=W[keep_inds,:,:,:]
            else:
                rec_weight=W[:,keep_inds,:,:]
        else:
            keep_inds_new=[]
            for i in keep_inds:
                l=i*X.shape[2]
                r=(i+1)*X.shape[2]
                for j in range(l,r):
                    keep_inds_new.append(j)
            rec_weight=W[:,keep_inds_new]

        return rec_weight
    def vgg_cifar_auto_prune_layer(self,alpha,model_name,threshold):
        self.pruned_model = model[model_name]()
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.pruned_model.load_state_dict(checkpoint)
        self.stayed_nums=[]
        for idx in self.prunable_idx:
            if idx == 2:
                continue
            X = self.layer_info_dict[idx]['input_feat']
            Y = self.layer_info_dict[idx]['output_feat']
            op = list(self.model.modules())[idx]

            W = op.weight.data.cpu().numpy()
            n, c = W.shape[0], W.shape[1]
            print(idx)
            keep_inds, keep_num = self.pruner(X, Y, W, alpha,threshold,debug=False)
            self.stayed_nums.append(int(keep_num))
            W_rec = self.prune_next_layer(X, Y,op, keep_inds, debug=False)  #当前层输入,要和前面对应上，inds*输入通道的一个集合
            # # assign new weight to pruned model
            self._prune_prev_layer(idx, W_rec, keep_inds)     #前一层输出
            self._record_pruning_layer(idx, op, c, keep_num)
        print(self.stayed_nums)
        X_model_name=model_name+"_X"
        tmp_model=model[X_model_name](self.stayed_nums).cuda()
        dummy = torch.rand((1, 3, 32, 32)).cuda()
        n_flops, n_params = profile(tmp_model, (dummy, ), verbose=False)
        return n_flops
    def resnet_cifar_auto_prune_layer(self,alpha,model_name,threshold):
        self.pruned_model = model[model_name]()
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.pruned_model.load_state_dict(checkpoint)
        self.stayed_nums=[]
        flag=1
        for idx in self.prunable_idx:
            if idx == 2:
                continue
            if flag == 1:
                op = list(self.model.modules())[idx]
                W = op.weight.data.cpu().numpy()
                n, c = W.shape[0], W.shape[1]
                self.stayed_nums.append(c)
                flag=0
                continue
            flag=1

            X = self.layer_info_dict[idx]['input_feat']
            Y = self.layer_info_dict[idx]['output_feat']
            op = list(self.model.modules())[idx]

            W = op.weight.data.cpu().numpy()
            n, c = W.shape[0], W.shape[1]
            print(idx)
            keep_inds, keep_num = self.pruner(X, Y, W, alpha,threshold,debug=False)
            self.stayed_nums.append(int(keep_num))
            W_rec = self.prune_next_layer(X, Y,op, keep_inds, debug=False)  #当前层输入,要和前面对应上，inds*输入通道的一个集合
            # # assign new weight to pruned model
            self._prune_prev_layer(idx, W_rec, keep_inds)     #前一层输出
            self._record_pruning_layer(idx, op, c, keep_num)
        print(self.stayed_nums)
        X_model_name=model_name+"_X"
        tmp_model=model[X_model_name](self.stayed_nums).cuda()
        dummy = torch.rand((1, 3, 32, 32)).cuda()
        n_flops, n_params = profile(tmp_model, (dummy, ), verbose=False)
        return n_flops
    def resnet_imagenet_auto_prune_layer(self,alpha,model_name,threshold):
        self.pruned_model = resnet50().cuda()
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items() if not k.endswith('total_params') and not k.endswith('total_ops')}
        self.pruned_model.load_state_dict(checkpoint)
        self.stayed_nums=[64]

        for (idx,name) in zip(self.prunable_idx,self.prunable_names):
            if idx == 2 or idx == 199:
                continue
            #conv1.conv直接跳过
            #conv2.conv 输入剪，上一层输出剪
            #conv3.conv输入剪，上一层输出剪
            #downsample 不剪，直接跳过
            if name.endswith('conv1.conv') or name.endswith('downsample.conv'):
                continue
            if name.endswith('conv2.conv'):
                X = self.layer_info_dict[idx]['input_feat']
                Y = self.layer_info_dict[idx]['output_feat']
            elif name.endswith('conv3.conv'):
                X = self.layer_info_dict[idx]['input_feat']
                next_id=idx+1
                while(next_id<=199):
                    name_next,op_next=list(self.model.named_modules())[next_id]
                    if name_next.endswith('conv_bn1.conv') or name_next.endswith('linear') or name_next.endswith('fc'):
                        Y = self.layer_info_dict[next_id]['input_feat']
                        Y=Y.reshape(Y.shape[0],-1)
                        break
                    next_id+=1
            op = list(self.model.modules())[idx]

            W = op.weight.data.cpu().numpy()
            n, c = W.shape[0], W.shape[1]
            print(idx)
            keep_inds, keep_num = self.pruner(X, Y, W, alpha,threshold,debug=False) #current layer input channels 
            self.stayed_nums.append(int(keep_num))
            W_rec = self.prune_next_layer(X, Y,op, keep_inds, debug=False)  #weights which pruned input channels
            # # assign new weight to pruned model
            self._prune_prev_layer(idx, W_rec, keep_inds)     #前一层输出
            self._record_pruning_layer(idx, op, c, keep_num)
        self.stayed_nums.insert(3, 256)
        self.stayed_nums.insert(10, 512)
        self.stayed_nums.insert(19, 1024)
        self.stayed_nums.insert(32, 2048)
        print(self.stayed_nums)
        tmp_model=resnet50_X(self.stayed_nums).cuda()
        dummy = torch.rand((1, 3, 224, 224)).cuda()
        n_flops, n_params = profile(tmp_model, (dummy, ), verbose=False)
        return n_flops
    
    def prune(self, ratios):
        for idx, ratio in ratios.items():
            print("pruning layer {}, pruning ratio {}".format(idx, ratio))
            self.prune_layer(idx, ratio)
    def metric(self, ):
        if self.val_dataloader is not None and self.criterion is not None:
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            self.pruned_model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.pruned_model(inputs)
                    loss = self.criterion(outputs, targets)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    progress_bar(batch_idx, len(self.val_dataloader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                                 .format(losses.avg, top1.avg, top5.avg))
   
    def googlenet_auto_prune_layer(self,alpha,model_name,threshold):
        self.pruned_model = googlenet().cuda()
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.pruned_model.load_state_dict(checkpoint)
        self.stayed_nums=[]
        for (idx,name) in zip(self.prunable_idx,self.prunable_names):
            if idx == 2 or idx == 251:
                continue
            if name.endswith('branch3x3.3') or name.endswith('branch5x5.3') or name.endswith('branch5x5.6'):
                #正常剪枝
                X = self.layer_info_dict[idx]['input_feat']
                Y = self.layer_info_dict[idx]['output_feat']
                op = list(self.model.modules())[idx]

                W = op.weight.data.cpu().numpy()
                n, c = W.shape[0], W.shape[1]
                print(idx)
                keep_inds, keep_num = self.pruner(X, Y, W, alpha,threshold,debug=False)
                #print(keep_inds)
                self.stayed_nums.append(int(keep_num))
                W_rec = self.prune_next_layer(X, Y,op, keep_inds, debug=False)  #当前层输入,要和前面对应上，inds*输入通道的一个集合
                # # assign new weight to pruned model
                self._prune_prev_layer(idx, W_rec, keep_inds)     #前一层输出
                self._record_pruning_layer(idx, op, c, keep_num)

            else:
                continue
        self.stayed_nums=np.array(self.stayed_nums).reshape(9,3).tolist()
        print(self.stayed_nums)
        tmp_model=googlenet_X(self.stayed_nums).cuda()
        dummy = torch.rand((1, 3, 32, 32)).cuda()
        n_flops, n_params = profile(tmp_model, (dummy, ), verbose=False)
        return n_flops
    def save_pruned_model(self, save_dir = None):
        if not os.path.exists(save_dir):
            print("dir {} does not exist".format(save_dir))
            return

        filename = os.path.join(save_dir, self.config.name + 'pruned.pth.tar')
        state_dict = {
            'state_dict': self.pruned_model.module.state_dict() \
                if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
            'pruning_info': self.pruning_info
        }
        torch.save(state_dict, filename)
    
    def auto_prune(self,name,alpha=1e-6,target_params=0,tolerance=0.01,threshold=1):
        left = 0  
        right = alpha
        lbound = target_params - tolerance * target_params
        rbound = target_params + tolerance * target_params

        while True:
            #传入α剪枝
            #prune(α)
            params=self.pruning_strategy[name](right,name,threshold)
            #求解参数量
            print("expected %d params, but got %d params" % (target_params, params))
            if params < target_params:  #参数量小于target
                    break
            else:
                    right *= 2

        # step=0
        while True:
            #step+=1
                # binary search
            alpha = (left + right) / 2
            params=self.pruning_strategy[name](alpha,name,threshold)  #剪枝，求参数量
            print('alpha: %.9f, params: %d, '
                  'left: %.9f, right: %.9f, left_bound: %.9f, right_bound: %.9f' %
                  (alpha,params , left, right, lbound, rbound))            
            if params > rbound:  #参数量大于target
                left=alpha
            elif params < lbound:  #参数量小于target
                right=alpha
            else:
                break

            if alpha < 1e-15:
                break
            # if step>50:
            #     break
