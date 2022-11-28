import torch
import torch.nn as nn
import math

norm_mean, norm_var = 0.0, 1.0

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, n5x5red2,pool_planes,tmp_name, compress_rate, last=False):
        super(Inception, self).__init__()
        self.tmp_name=tmp_name

        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            conv1x1 = nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name

            self.branch1x1 = nn.Sequential(
                conv1x1,
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:

            if last:
                output=n3x3
            else:
                output=math.ceil(round(n3x3*(1.-compress_rate[0]),3))
            # print('-',n3x3,cp_rate,round(n3x3*cp_rate,3),output,int(n3x3*(1.0-0.8)))

            conv3x3_1=nn.Conv2d(in_planes, n3x3red, kernel_size=1)
            conv3x3_2=nn.Conv2d(n3x3red, output, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name

            self.branch3x3 = nn.Sequential(
                conv3x3_1,
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                conv3x3_2,
                nn.BatchNorm2d(output),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:

            if last:
                output=n5x5
            else:
                output=math.ceil(round(n5x5*(1.-compress_rate[2]),3))

            conv5x5_1 = nn.Conv2d(in_planes, n5x5red, kernel_size=1)
            conv5x5_2 = nn.Conv2d(n5x5red, n5x5red2, kernel_size=3, padding=1)
            conv5x5_3 = nn.Conv2d(n5x5red2, output, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name

            self.branch5x5 = nn.Sequential(
                conv5x5_1,
                nn.BatchNorm2d(n5x5red),
                nn.ReLU(True),
                conv5x5_2,
                nn.BatchNorm2d(n5x5red2),
                nn.ReLU(True),
                conv5x5_3,
                nn.BatchNorm2d(output),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            conv_pool = nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name

            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                conv_pool,
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        y1 = self.branch1x1(x)
        out.append(y1)

        y2 = self.branch3x3(x)
        out.append(y2)

        y3 = self.branch5x5(x)
        out.append(y3)

        y4 = self.branch_pool(x)
        out.append(y4)
        return torch.cat(out, 1)

class GoogLeNet(nn.Module):
    def __init__(self, compress_rate=None, block=Inception, filters=None,cfg=None):
        super(GoogLeNet, self).__init__()

        first_outplanes=math.ceil(192*(1.-compress_rate[0]))
        conv_pre = nn.Conv2d(3, first_outplanes, kernel_size=3, padding=1)
        conv_pre.tmp_name = 'pre_layer'
        self.pre_layers = nn.Sequential(
            conv_pre,
            nn.BatchNorm2d(first_outplanes),
            nn.ReLU(True),
        )

        filters = [
            [64, 128, 32, 32],
            [128, 192, 96, 64],
            [192, 208, 48, 64],
            [160, 224, 64, 64],
            [128, 256, 64, 64],
            [112, 288, 64, 64],
            [256, 320, 128, 128],
            [256, 320, 128, 128],
            [384, 384, 128, 128]
        ]
        self.filters = filters
        if cfg == None:
            mid_filters = [
                [96, 16,32],
                [128, 32,96],
                [96, 16,48],
                [112, 24,64],
                [128, 24,64],
                [144, 32,64],
                [160, 32,128],
                [160, 32,128],
                [192, 48,128]
            ]
        else:
            mid_filters = cfg

        self.mid_filters = mid_filters

        in_plane_list=[]
        for i in range(8):
            in_plane_list.append(filters[i][0]+math.ceil(round(filters[i][1]*(1.-compress_rate[i*3+1]) ,3) )+math.ceil(round(filters[i][2]*(1.-compress_rate[i*3+3]) ,3))+filters[i][3])

        self.inception_a3 = block(first_outplanes, filters[0][0], mid_filters[0][0], filters[0][1], mid_filters[0][1], filters[0][2], mid_filters[0][2],filters[0][3], 'a3', compress_rate[1:3+1])
        self.inception_b3 = block(in_plane_list[0], filters[1][0], mid_filters[1][0], filters[1][1], mid_filters[1][1], filters[1][2],mid_filters[1][2], filters[1][3], 'a4', compress_rate[3+1:3*2+1])

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_a4 = block(in_plane_list[1], filters[2][0], mid_filters[2][0], filters[2][1], mid_filters[2][1], filters[2][2],mid_filters[2][2], filters[2][3], 'a4', compress_rate[3*2+1:3*3+1])
        self.inception_b4 = block(in_plane_list[2], filters[3][0], mid_filters[3][0], filters[3][1], mid_filters[3][1], filters[3][2],mid_filters[3][2], filters[3][3], 'b4', compress_rate[3*3+1:3*4+1])
        self.inception_c4 = block(in_plane_list[3], filters[4][0], mid_filters[4][0], filters[4][1], mid_filters[4][1], filters[4][2],mid_filters[4][2], filters[4][3], 'c4', compress_rate[3*4+1:3*5+1])
        self.inception_d4 = block(in_plane_list[4], filters[5][0], mid_filters[5][0], filters[5][1], mid_filters[5][1], filters[5][2],mid_filters[5][2], filters[5][3], 'd4', compress_rate[3*5+1:3*6+1])
        self.inception_e4 = block(in_plane_list[5], filters[6][0], mid_filters[6][0], filters[6][1], mid_filters[6][1], filters[6][2],mid_filters[6][2], filters[6][3], 'e4', compress_rate[3*6+1:3*7+1])

        self.inception_a5 = block(in_plane_list[6], filters[7][0], mid_filters[7][0], filters[7][1], mid_filters[7][1], filters[7][2],mid_filters[7][2], filters[7][3], 'a5', compress_rate[3*7+1:3*8+1])
        self.inception_b5 = block(in_plane_list[7], filters[8][0], mid_filters[8][0], filters[8][1], mid_filters[8][1], filters[8][2],mid_filters[8][2], filters[8][3], 'b5', compress_rate[3*8+1:3*9+1], last=True)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(sum(filters[-1]), 10)

    def forward(self, x):

        out = self.pre_layers(x)

        # 192 x 32 x 32
        out = self.inception_a3(out)
        # 256 x 32 x 32
        out = self.inception_b3(out)
        # 480 x 32 x 32
        out = self.maxpool1(out)

        # 480 x 16 x 16
        out = self.inception_a4(out)
        # 512 x 16 x 16
        out = self.inception_b4(out)
        # 512 x 16 x 16
        out = self.inception_c4(out)
        # 512 x 16 x 16
        out = self.inception_d4(out)
        # 528 x 16 x 16
        out = self.inception_e4(out)
        # 823 x 16 x 16
        out = self.maxpool2(out)

        # 823 x 8 x 8
        out = self.inception_a5(out)
        # 823 x 8 x 8
        out = self.inception_b5(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def googlenet(compress_rate = None,oristate_dict = None,ranks = None):
    model = None
    if compress_rate is None :
        compress_rate = [0.]*28
        model = GoogLeNet(compress_rate=compress_rate, block=Inception)
    elif oristate_dict is not None and ranks is not None :
        model = GoogLeNet(compress_rate=compress_rate, block=Inception)
        state_dict = model.state_dict()

        last_select_index = None
        mid_filters = model.mid_filters
        filters = model.filters

        _in_rank    = ranks[0][0]
        _rank       = []
        _3x3_rank   = ranks[1][0]
        _5x5_rank_1 = ranks[1][1]
        _5x5_rank_2 = ranks[1][2]

        base   = 7
        _base  = 0
        ind    = 0 # index in mid_filters and filters
        slices = [6,55,104,153,202,251,300,349,398,447]
        cnt    = 1

        for k,name in enumerate(oristate_dict):
            if k < 7:
                if k >= 0 and k <=5 :
                    for index_i,i in enumerate(_in_rank):
                        state_dict[name][index_i] = oristate_dict[name][i]
                else:
                    state_dict[name] = oristate_dict[name]
            elif k <= 447:
                _k = k - base
                if _k % 7 == 0 :
                    _ = int(_k/7)
                    if _ == 0:
                        for _i,i in enumerate(list(range(filters[ind][0]))):
                            for _j,j in enumerate(_in_rank):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                        l      = filters[ind][0]
                        rank   = list(range(l))
                        _rank += [x + _base for x in rank]
                        _base += l
                    elif _ == 6:
                        for _i,i in enumerate(list(range(filters[ind][3]))):
                            for _j,j in enumerate(_in_rank):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                        l      = filters[ind][3]
                        rank   = list(range(l))
                        _rank += [x + _base for x in rank]
                        _base += l
                    elif _ == 1:
                        for _i,i in enumerate(list(range(mid_filters[ind][0]))):
                            for _j,j in enumerate(_in_rank):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    elif _ == 3:
                        for _i,i in enumerate(list(range(mid_filters[ind][1]))):
                            for _j,j in enumerate(_in_rank):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    elif _ == 2 :
                        for _i,i in enumerate(_3x3_rank):
                            for _j,j in enumerate(range(mid_filters[ind][0])): # todo
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                        _rank += [x + _base for x in _3x3_rank]
                        _base += len(oristate_dict[name])
                    elif _ == 4 :
                        for _i,i in enumerate(_5x5_rank_1):
                            for _j,j in enumerate(range(mid_filters[ind][1])):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                    elif _ == 5:
                        for _i,i in enumerate(_5x5_rank_2):
                            for _j,j in enumerate(_5x5_rank_1):
                                state_dict[name][_i][_j] = oristate_dict[name][i][j]
                        _rank += [x + _base for x in _5x5_rank_2]
                        _base += len(oristate_dict[name])
                elif _k % 7 == 6:
                    state_dict[name] = oristate_dict[name]
                elif _k % 7 > 0 and _k % 7 < 6 :
                    _ = int(_k/7)
                    if _ in [0,1,3,6]:
                        state_dict[name] = oristate_dict[name]
                    elif _ == 2 :
                        for _i,i in enumerate(_3x3_rank):
                            state_dict[name][_i] = oristate_dict[name][i]
                    elif _ == 4  :
                        for _i,i in enumerate(_5x5_rank_1):
                            state_dict[name][_i] = oristate_dict[name][i]
                    elif _ == 5:
                        for _i,i in enumerate(_5x5_rank_2):
                            state_dict[name][_i] = oristate_dict[name][i]

                if k == slices[cnt]:
                    base = slices[cnt] + 1
                    _in_rank = _rank
                    if cnt >= 9 : continue
                    _3x3_rank   = ranks[cnt+1][0]
                    _5x5_rank_1 = ranks[cnt+1][1]
                    _5x5_rank_2 = ranks[cnt+1][2]
                    _rank = []
                    _base = 0
                    cnt  += 1
                    ind   = cnt -1 
            elif k == 448:
                for _i,i in enumerate(range(10)):
                    for _j,j in enumerate(_in_rank):
                        state_dict[name][_i][_j] = oristate_dict[name][i][j]
            elif k == 449:
                state_dict[name] = oristate_dict[name]
        model.load_state_dict(state_dict)
    else:
        model = GoogLeNet(compress_rate=compress_rate, block=Inception)
    return model
def googlenet_X(cfg):
    compress_rate = [0.]*28
    return GoogLeNet(cfg=cfg,compress_rate=compress_rate)
