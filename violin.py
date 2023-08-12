import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.io import loadmat
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']  # 解决matplotlib无法显示中文的问题
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题

#violin作用就是用来显示数据的分布和概率密度，可以看成是箱线图和密度图的结合。中间部分反映的是箱线图的信息，图的两侧反映的是密度图的信息。
#小提琴图中间的黑色粗条用来显示四分位数。黑色粗条中间的白点表示中位数，粗条的顶边和底边分别表示上四分位数和下四分位数，通过边的位置所对应的y轴的数值就可以看到四分位数的值。
#由黑色粗条延伸出的黑细线表示95%的置信区间。

# df = pd.read_excel('F:/SSA-LSTM/Data/comparison/hour1/SSA-LSTM(gru3)_result_hour1(3).xl')
# df = pd.read_excel(r'F:\SSA-LSTM\Data\SSA-LSTM(gru3)_result_hour1.xls', index_col=0)
# data8=loadmat(r'F:\SSA-LSTM\Data\result\hour1\bpred.mat')['b']
# data8=data8.tolist()
df = pd.read_excel(r'.\violin2.xls', index_col=0)
# x=df['ph'].value_counts()

sns.violinplot(x=df['Component'],
               y=df['Prediction'])
                  # data=df,
                  # order=['1','2','12','24'])
#plt.title('DO prediction under different hours')
plt.xlabel('Components',fontsize=17)
plt.ylabel('Prediction',fontsize=17)
plt.legend(loc='upper right')
plt.savefig(r'.\violin2.png')
plt.show()