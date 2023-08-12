import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib as mpl
from numpy import polyfit, poly1d
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']  # 解决matplotlib无法显示中文的问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题
plt.rcParams['font.sans-serif'] = 'Times New Roman'
def result(pred,real):
    # mape
    mape=np.mean(np.abs((pred-real)/real))
    # rmse
    rmse=np.sqrt(np.mean(np.square(pred-real)))
    # mae
    mae=np.mean(np.abs(pred-real))
    # R2
    r2=r2_score(real,pred)

    ave_real=np.sum(real)
    # fenzi=np.sum(math.pow(real-pred,2))
    # fenmu=np.sum(math.pow(real-ave_real,2))
    # nse=1-fenzi/fenmu
    ave_real=np.mean(real)
    fenzi=np.sum(np.square(pred-real))
    fenmu=np.sum(np.square(real - ave_real))
    nse=1-fenzi/fenmu

    sse=np.sum(np.square(pred-real))
    return mape,rmse,mae,r2,nse,sse


def R(pred,real):
    r=np.corrcoef(pred,real)[0,1]
    return r

alnbsa_true=loadmat(r'.\result\ALN-LSTM2(r).mat')['true']
alnbsa_pred=loadmat(r'.\result\ALN-LSTM2(r).mat')['pred']



alnbsa_mape1,alnbsar_mse1,alnbsa_mae1,alnbsa_r21,alnbsa_nse1,alnbsa_sse1=result(alnbsa_pred,alnbsa_true)

print('alnbsa的mape:',alnbsa_mape1,' rmse:',alnbsar_mse1,' mae:',alnbsa_mae1,' R2:',alnbsa_r21,' NSE:',alnbsa_nse1,' SSE:',alnbsa_sse1)

xx=alnbsa_true.flatten()
coeff=polyfit(xx,alnbsa_pred,1)
print("coeff:",coeff)

r=R(alnbsa_pred,alnbsa_true)
std=np.std(alnbsa_pred,ddof=1)
print("std:",std, 'R:',coeff)


plt.figure(figsize=(8,6))
# plt.xlabel("Raw data",fontsize=14)
# plt.ylabel("Predictive value",fontsize=14)
plt.xlabel("Raw remainder data(mm)",fontsize=14)
plt.ylabel("Predicted remainder value(mm)",fontsize=14)
plt.scatter(alnbsa_true,alnbsa_pred,c='r',marker='.',alpha=0.4,label='Sample data')
plt.plot(xx,coeff[0]*xx+coeff[1],color='blue',label='Fit curve')
plt.title("Correlation coefficient: R=0.5243",fontsize=14)
plt.legend(loc='upper left', fontsize=13)
plt.savefig('.\picture\corr(r).png',dpi=1000,bbox_inches = 'tight')
plt.show()
