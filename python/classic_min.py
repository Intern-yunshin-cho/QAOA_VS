import numpy as np 
'''
    classic Method : AIC, CN
'''
def mse(X,y):
    q = X.shape[0]
    xtx = np.matmul(X.T,X)
    xtxinv = np.linalg.inv(xtx)
    proj = np.matmul(np.matmul(X,xtxinv),X.T)
    mse = np.matmul(np.matmul(y.T,(np.identity(q)-proj)),y)/q
    return(mse)

def cn(X): 
    if X.shape[1]==1 : return(1)
    cordata = np.corrcoef(X.T)
    eig_value = np.linalg.eig(cordata)[0]
    return(np.max(eig_value)/np.min(eig_value))


def classics(X,y):
    n,p = X.shape[0], X.shape[1]
    mse_list = []
    rsquare_list =[]
    aic_list = []
    cn_list = []
    for i in range(1,2**p):
        binind = [bool(int(j)) for j in bin(i)[2:]]
        binind = [False for i in range(p-len(binind))]+ binind
        X_choose = X[:,binind]
        q = X_choose.shape[1]
        mse_temp = mse(X_choose,y)
        cn_list  += [cn(X_choose)]
        mse_list += [mse_temp]
        rsquare_list += [1-mse_temp/np.var(y)]
        aic_list += [2*q+n*np.log(mse_temp)]

    return mse_list, rsquare_list, aic_list, cn_list

def qaoa_ind(X,y,result):
    n,p = X.shape[0], X.shape[1]
    qaoa_ind = [bool(i) for i in result]
    X_choose = X[:,qaoa_ind]
    mse_qaoa = mse(X_choose,y)
    aic_qaoa = 2*sum(qaoa_ind)+n*np.log(mse_qaoa)
    rsquare_qaoa = 1-mse_qaoa/np.var(y)
    cn_qaoa  = cn(X_choose)
    return aic_qaoa, cn_qaoa, mse_qaoa, rsquare_qaoa