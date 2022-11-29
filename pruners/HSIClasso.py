from re import S
import numpy as np
import time
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import os
import logging
#import cuml
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device(f"cuda:{1}") if torch.cuda.is_available() else 'cpu'
from sklearn.metrics.pairwise import rbf_kernel
def centering(K):
    if len(K.shape)==3:
        d = K.shape[0]
        n = K.shape[1]
        I=torch.randn((d,n,n))
        I[:]=torch.eye(n)
        unit = torch.ones((d,n,n))
        H = I - unit / n
        return torch.matmul(torch.matmul(H,K),H)
    else :
        n=K.shape[0]
        I=torch.eye(n)
        unit=torch.ones((n,n))
        H = I - unit / n
        return torch.matmul(torch.matmul(H,K),H)

def HSIC_lasso_pruning(X, Y, W, alpha=1e-3, threshold=1,debug=False):
    # Example: B is sample number
    #        : c_in is channel input
    #        : c_out is channel output
    #        : 3x3 is kernel size
    # X shape: [B, c_in, 3, 3]
    # Y shape: [B, c_out]
    # W shape: [c_out, c_in, 3, 3]
    # Linear
    # X shape: [B, c_in]
    # Y shape: [B, c_out]
    # W shape: [c_out, c_in]
    ##求HSIC X(B,c_in,h*w)  Y(B,c_out*h*w)
    b, c, l = X.shape
    X=X.transpose(1,0,2)  #(c_in,B,h*w)
    K=None
    X=torch.from_numpy(X)#.to(device)
    #K=X.matmul(torch.from_numpy(X.numpy().transpose(0,2,1)))
    tmp = X.unsqueeze(2) - X.unsqueeze(1) # (c_in, B,B, h*w)
    gamma = 1.0 / X.shape[-1]
    K = torch.exp(- (tmp ** 2).sum(dim=-1) * gamma) #(c_in,N,N)
    K_ba=centering(K) #(c_in,N,N)
    K_ba=K_ba.reshape(c,-1)
    K_ba=K_ba.transpose(1,0).numpy()# (N*N,c_in)

        

    Y=torch.from_numpy(Y)
    tmp=Y.unsqueeze(1) - Y.unsqueeze(0)
    gamma = 1.0 / Y.shape[-1]
    L = torch.exp(- (tmp ** 2).sum(dim=-1) * gamma)
    L_ba=centering(L)
    L_ba=L_ba.reshape(-1).numpy()#N*N

    solver = Lasso(alpha=alpha, warm_start=True, selection='random', random_state=0)
    def solve(alpha):
        """ Solve the Lasso"""
        solver.alpha = alpha
        solver.fit(K_ba, L_ba)
        nonzero_inds = np.where(solver.coef_ != 0.)[0]
        nonzero_num = sum(solver.coef_ != 0.)
        return nonzero_inds, nonzero_num, solver.coef_



    left = 0  
    right = alpha
    #while True:
    keep_inds, keep_num, coef = solve(right)

    if keep_num<threshold:
        lbound=threshold
        rbound=threshold
        step=0
        while True:
            step+=1
            # binary search
            alpha = (left + right) / 2
            keep_inds, keep_num, coef = solve(alpha)
            # print loss
            # product has size [B x c_out, c_in]
            loss = 1 / (2 * float(X.shape[0])) * \
                np.sqrt(np.sum((L_ba - np.matmul(K_ba, coef)) ** 2, axis=0)) + \
                alpha * np.sum(np.fabs(coef))

            if debug:
                print('loss: %.6f, alpha: %.6f, feature nums: %d, '
                    'left: %.6f, right: %.6f, left_bound: %.6f, right_bound: %.6f' %
                    (loss, alpha, keep_num, left, right, lbound, rbound))

            if keep_num > rbound:
                left=alpha
            elif keep_num < lbound:
                right=alpha
            else:
                break

            if alpha < 1e-15:
                break
            if step>50:
                break
    print("orig chn num = {} keep chn num = {}".format(c, keep_num))
    logging.info("orig chn num = {} keep chn num = {}".format(c, keep_num))
    return keep_inds, keep_num

if __name__ == '__main__':
    X=torch.randn((6,2,6)).numpy()
    Y=torch.randn((6,40)).numpy()
    HSIC_lasso_pruning(X,Y,None,5)