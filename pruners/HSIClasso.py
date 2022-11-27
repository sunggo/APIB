from re import S
import numpy as np
import time
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import os
import logging
import time
#import cuml
os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device(f"cuda:{2}") if torch.cuda.is_available() else 'cpu'
from sklearn.metrics.pairwise import rbf_kernel
def centering(K):
    if len(K.shape)==3:
        d = K.shape[0]
        n = K.shape[1]
        I = np.tile(np.eye(n), (d, 1, 1))
        # I=torch.randn((d,n,n))
        # I[:]=torch.eye(n)
        # unit = np.full((d,n,n), 1/n)
        H = I - np.full((d,n,n), 1/n) #unit / n
        return H@K@H
    else :
        n=K.shape[0]
        I = np.eye(n)
        # I=torch.eye(n)
        # unit=torch.ones((n,n))
        H = I - np.full((n,n), 1/n)#unit / n
        return H@K@H

def HSIC_lasso_pruning(X, Y, W, alpha=1e-6, threshold=1,debug=False):
    
    # Conv
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
    # X=torch.from_numpy(X)        
    #K=X.matmul(torch.from_numpy(X.numpy().transpose(0,2,1)))#linear kernel
    G = X@X.transpose(0,2,1)
    H = np.tile(np.diagonal(G, axis1=1,axis2=2)[:,None,:], (1,b, 1))
    gamma = 1.0 / X.shape[-1]
    K = np.exp((H + H.transpose(0,2,1) -2*G)*(-gamma))
    # tmp = X.unsqueeze(2) - X.unsqueeze(1) # (c_in, B,B, h*w)
    # K = torch.exp(- (tmp ** 2).sum(dim=-1) * gamma) #(c_in,N,N)
    
    K_ba=centering(K) #(c_in,N,N)
    K_ba=K_ba.reshape(c,-1)
    K_ba = K_ba.T
    # K_ba=K_ba.transpose(1,0).numpy()# (N*N,c_in)

    
        
    Y=Y-Y.mean(0)
    # Y=torch.from_numpy(Y)
    G = Y@Y.T
    H = np.tile(np.diagonal(G), (b, 1))
    gamma = 1.0 / Y.shape[-1]
    L = np.exp((H + H.T -2*G)*(-gamma))  
    
    #L=Y.matmul(Y.transpose(1,0))
    # tmp=Y.unsqueeze(1) - Y.unsqueeze(0)
    # gamma = 1.0 / Y.shape[-1]
    # L = torch.exp(- (tmp ** 2).sum(dim=-1) * gamma)
    
    L_ba=centering(L)
    L_ba=L_ba.reshape(-1)#.numpy()#N*N

    # use LassoLars because it's more robust than Lasso
    solver = Lasso(alpha=alpha, warm_start=True, selection='random', random_state=0)

    def solve(alpha):
        """ Solve the Lasso"""
        solver.alpha = alpha
        solver.fit(K_ba, L_ba)
        nonzero_inds = np.where(solver.coef_ != 0.)[0]
        nonzero_num = sum(solver.coef_ != 0.)
        # np.isclose(solver.coef_,0, atol=1e-7)
        return nonzero_inds, nonzero_num, solver.coef_

    tic = time.perf_counter()


    left = 0  # minimum alpha is 0, which means don't use lasso regularizer at all
    right = alpha
    keep_inds, keep_num, coef = solve(right)
    if keep_num < threshold:
        lbound=threshold   # threshold
        rbound=threshold   # threshold
        step=0
        while True:
            step+=1
            # binary search
            alpha = (left + right) / 2
            keep_inds, keep_num, coef = solve(alpha)
            # print loss
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

            if alpha < 1e-20:
                break
            if step>50:
                break

        toc = time.perf_counter()
    print("orig chn num = {} keep chn num = {}".format(c, keep_num))
    logging.info("orig chn num = {} keep chn num = {}".format(c, keep_num))
    return keep_inds, keep_num

if __name__ == '__main__':
    X=torch.randn((6,2,6)).numpy()
    Y=torch.randn((6,40)).numpy()
    HSIC_lasso_pruning(X,Y,None,5)