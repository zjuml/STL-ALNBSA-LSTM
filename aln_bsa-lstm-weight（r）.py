#ALNBSA优化lstm权值和阈值
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from sklearn.metrics import r2_score
from scipy.io import savemat
import matplotlib as mpl
import random
import time
import random as rn
mpl.rcParams['font.serif'] = ['KaiTi']  # 解决matplotlib无法显示中文的问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def build_timeseries(dataset, target_column_index, time_steps):
    # Total number of time-series samples would be len(dataset) - time_steps
    time_steps=int(time_steps)
    dim_0 = int(dataset.shape[0] - time_steps ) # 样本个数

    print(dim_0)
    dim_1 = dataset.shape[1]  # 维度，即特征向量个数
    print(dim_1)
    print('---------------',dim_0,int(time_steps),dim_1)
    # Init the x, y matrix,
    x = np.zeros((dim_0, int(time_steps), dim_1))  
    # Number of output variables is 1 for this mid price prediction
    y = np.zeros((dim_0, 1))  

    # fill data to x, y
    for i in range(dim_0):
        x[i] = dataset[i: i + time_steps]
        y[i] = dataset[i + time_steps, target_column_index]

    print("Lenght of time-series x, y", x.shape, y.shape)
    return x, y

df = pd.read_excel('./data/remainder.xls', index_col=0)
values = df.values
# valuess=df.values
# print("values:",values[0:5])
# encoder = LabelEncoder()
# values[:, 4] = encoder.fit_transform(values[:, 4])
# print("values:",values[0:5])
values = values.astype('float32')
# values2=values[:,2:3]#trend
# values2=values[:,3:4]#resid
print("values.shape:",values.shape)
# print("values2.shape:",values2.shape)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[12,13,14,15,16,17,18,19,20,21]], axis=1, inplace=True)
print("reframed.shape:",reframed.shape)
values = reframed.values

m = int(values.shape[0] * 0.2)  # 训练集：测试集=8:2
train, test = values[0:-m, :], values[-m:, :]
print("train:",train.shape)
print("test:",test.shape)

target_column_index = 0 
lookback_window = 7  

train_X, train_y = build_timeseries(train, target_column_index,lookback_window)  
test_X, test_y = build_timeseries(test, target_column_index,lookback_window)  
train_X = train_X[:, :, :-1]  
train_y = train_y[:, -1:]  
test_X = test_X[:, :, :-1]  
test_y = test_y[:, -1:]  
print("train_data.shape", train_X.shape)
print("train_label.shape", train_y.shape)
print("test_data.shape", test_X.shape)
print("test_label.shape", test_y.shape)

def reshapeX(x,dir):#reshape作用就是将xleft变成weight一样矩阵的形式
    b=[]
    k=0
    for i in range(x.size):#x.size指的就是网络权重的个数
        c=np.array(dir[k:k+x[i].size])
        a=c.reshape(x[i].shape)
        b.append(a)
        k = k +x[i].size
    return np.array(b)

def stretch(weight):
    b=[]
    b=np.array(b)
    k=0
    for i in range(weight.size):
        k=k+weight[i].size
        a=weight[i].reshape(-1)
        a=np.array(a)
        b=np.hstack((b,a))
    b= b.reshape(-1)
    return b

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(4,activation='tanh'))#relu,sigmoid
model.add(Dense(1,activation='tanh'))
model.compile(loss='mae', optimizer='adam')

weight=np.array(model.get_weights())
"""
smell=model.evaluate(test_X,test_y,1)
print(smell)
"""
dim=0
for i in range(weight.size):
    dim=weight[i].size+dim #空间维度

#  Population generation
def population_generation(low, high, pop_size):
  population = np.random.uniform(low, high, pop_size)
  return population
# Boundary control
# def Boundary_control(offsprings):
#     offsprings[offsprings<low] = low
#   for i in range(pop_number):
#     for j in range(dim):
#       k = rn.random() < rn.random()
#       if offsprings[i][j] < low:
#         # offers[offers<low] = low
#         if k:
#           offsprings[i][j] = low
#         else:
#           offsprings[i][j] = np.random.uniform(low, high)
#       if offsprings[i][j] > high:
#         if k:
#           offsprings[i][j] = high
#         else:
#           offsprings[i][j] = np.random.uniform(low, high)
#   return offsprings

# 替换上述boundary_control代码
def Boundary_control(offsprings, low, high):
    offsprings = np.clip(offsprings, low, high)
    mask_low = offsprings < low
    mask_high = offsprings > high
    k = np.random.rand(*offsprings.shape) < np.random.rand()
    offsprings[mask_low & k] = low
    offsprings[mask_low & ~k] = np.random.uniform(low, high, size=np.sum(mask_low & ~k))
    offsprings[mask_high & k] = high
    offsprings[mask_high & ~k] = np.random.uniform(low, high, size=np.sum(mask_high & ~k))
    return offsprings

# BSA 
pop_number = 50
dim_rate=1
pop_size=(pop_number,dim)
low=-1
high=1
# pop= np.zeros((pop_number, dim))#种群初始化
# historical_pop= np.zeros((pop_number, dim))
epoch = 400
globalminimum =np.zeros(epoch)#存放每次迭代最小值
fitness_pop = np.zeros((pop_number, 1))  # pop行1列，为每个个体的适应度函数值
fitness_historical_pop = np.zeros((pop_number, 1))  # pop行1列，为每个个体的适应度函数值
fitness_offsprings = np.zeros((pop_number, 1))  # 突变交叉之后种群的适应度值，pop行1列，为每个个体的适应度函数值
indices= np.zeros((pop_number, 1))
# sum_fitness_pop=0
# sum_fitness_historical_pop=0
#种群初始化
pop=population_generation(low, high, pop_size)
historical_pop = population_generation(low, high, pop_size)
d=np.zeros((pop_number-1, pop_number-1), dtype=float)
L=np.zeros((pop_number-1, 1))
R=np.zeros((pop_number-1, pop_number-1), dtype=float)
nfmin=1
nfmax=4
#求pop的适应度值
for i in range(pop_number):  # 每个种群
    # pop[i, :] = np.random.rand(1,dim)#生成[1,dim]的矩阵，且在（0,1）之间，不包括1
    # historical_pop[i,:]=np.random.rand(1,dim)
    ww = reshapeX(weight, pop[i, :])
    model.set_weights(ww)
    # 计算适应度值
    fitness_pop[i] = model.evaluate(train_X, train_y, batch_size=128)
#     sum_fitness_pop=sum_fitness_pop+fitness_pop[i]
# avg_fitness_pop=1.0*sum_fitness_pop/pop_number
fMin=np.min(fitness_pop)
bestI = np.argmin(fitness_pop)  # 全局最优适应度值的下标bestI
bestX = pop[bestI, :].copy()  ##通过bestI，再找到全局最优个体位


#求pop的适应度值1
for i in range(pop_number):  # 每个种群
    # pop[i, :] = np.random.rand(1,dim)#生成[1,dim]的矩阵，且在（0,1）之间，不包括1
    # historical_pop[i,:]=np.random.rand(1,dim)
    ww = reshapeX(weight, historical_pop[i, :])
    model.set_weights(ww)
    # 计算适应度值
    fitness_historical_pop[i] = model.evaluate(train_X, train_y, batch_size=128)
#     sum_fitness_historical_pop=sum_fitness_historical_pop+fitness_historical_pop[i]
# avg__fitness_historical_pop=1.0*sum_fitness_historical_pop/pop_number

for epk in range(epoch):
    print("现在epk=:", epk)
    # Selection 1
    A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1  # 1*dim矩阵，随机分配1和-1值
    # L=np.ones((1, dim))
    a=0.35
    b=0.65
    n=3
    # F = 3 * np.random.randn()#标准正态分布（均值0，标准差1）
    dim_rate = 1 - 0.5 * (1 - rn.random())
    avg_fitness_pop=np.mean(fitness_pop)
    avg__fitness_historical_pop=np.mean(fitness_historical_pop)
    Ratio=1.0*avg_fitness_pop/(avg_fitness_pop+avg__fitness_historical_pop)

    if rn.random() < Ratio:
        historical_pop = pop
    historical_pop = historical_pop[np.random.permutation(pop_number), :]
    #求fitness_historical_pop
    for i in range(pop_number):  # 每个种群
        # pop[i, :] = np.random.rand(1,dim)#生成[1,dim]的矩阵，且在（0,1）之间，不包括1
        # historical_pop[i,:]=np.random.rand(1,dim)
        ww = reshapeX(weight, historical_pop[i, :])
        model.set_weights(ww)
        # 计算适应度值
        fitness_historical_pop[i] = model.evaluate(train_X, train_y, batch_size=128)
    map = np.zeros((pop_number, dim), dtype=int)
    if rn.random() < rn.random():
        for i in range(pop_number):
            u = np.random.permutation(dim)
            map[i][u[1:math.ceil(dim_rate * rn.random() * dim)]] = 1
    else:
        for i in range(pop_number):
            map[i][rn.randrange(1, dim + 1) - 1] = 1

    # Recombination (Mutation + Crossover)#突变交叉成新的种群
    #F=n*np.random.randn()*(1.0/(1+np.exp(epk*1.0/epoch)))##np.random.randn()是标准正态分布（均值0，标准差1）
    F=n*(1.0/(1+np.exp(epk*1.0/epoch)))

    #offsprings = pop + (map * F) * (a*(historical_pop - pop)+b*(bestX-pop))
    offsprings = pop + (map * F) * (a*(historical_pop - pop)+b*(bestX-pop)).dot(A.T * (A * A.T) ** (-1)) * np.ones((1, dim))
    print("offsprings:", offsprings)
    # Call boundary control
    offsprings = Boundary_control(offsprings,low,high)
    print("F:", F)
    print("map:", map)
    print("pop:", pop)
    print("offsprings:",offsprings)



    # Selection 2
    for i in range(pop_number):
        a = reshapeX(weight, offsprings[i, :])
        model.set_weights(a)
        fitness_offsprings[i] = model.evaluate(train_X, train_y, batch_size=64)  # 突变交叉之后的适应度值
    #     if fitness_offsprings[i] < fitness_pop[i]:
    #         fitness_pop[i]=fitness_offsprings[i]
    #         pop[i]=offsprings[i]
    # for i in range(pop_number):
    #     if fitness_pop[i]<fMin:
    #         fMin=fitness_pop[i]
    #         bestX=pop[i,:]
    # globalminimum[epk]=fMin
    # print("迭代次数，最优适应度值，最优参数组合分别是:",epk,fMin,bestX)

    #求距离
    # for i in range(pop_number-1):
    #     for j in range(i+1,pop_number):
    #         for k in range(dim):
    #             #计算pop种群之间的距离
    #             d[i][j-1]=np.sqrt(np.sum((pop[i][k]-pop[j][k])**2))
    #             d[j-1][i]=d[i][j-1]

    #             #计算pop与新种群T之间的距离
    #             R[i][j - 1] = np.sqrt(np.sum((pop[i][k] - offsprings[j][k]) ** 2))
    #             R[j - 1][i] = R[i][j - 1]
    #     L[i]=np.min(d[i, :])#计算d中每一行的最小值：每个个体与其他个体距离的最小值
    # print("d:",d)
    # print("L:",L)
    # nf=nfmin+(nfmax-nfmin)*math.asin(epk/epoch)
    # NR=np.sum(L)/(nf*(pop_number-1)+rn.random())#半径

    #上述代码简化 （避免使用嵌套循环）
    dd=np.sum(pop**2,axis=1)
    d= np.sqrt(dd[:, np.newaxis] + dd - 2*np.dot(pop, pop.T))#计算pop种群之间距离
    L= np.amin(d, axis=1)#计算d中每行最小值：每个个体与其他个体距离的最小值
    R=np.linalg.norm(pop - offsprings, axis=1)#计算pop与新种群T之间的距离
    nf=nfmin+(nfmax-nfmin)*math.asin(epk/epoch)
    NR=np.sum(L)/(nf*(pop_number-1)+rn.random())#半径
    print("R:",R)
    print("NR:",NR)


    #惩罚
    for i in range(pop_number-1):
        # for j in range(pop_number-1):
            # if R[i][j]<NR:
            if R[i]<NR:
                if fitness_pop[i]>fitness_offsprings[i]:
                    fitness_pop[i]=fitness_offsprings[i]+1# 1是惩罚项C
                if fitness_pop[i]<=fitness_offsprings[i]:
                    fitness_offsprings[i]=fitness_offsprings[i]+1

#         indices[i]=fitness_offsprings[i] - fitness_pop[i]
#     # a = reshapeX(weight, offsprings[i,:])
#     # print("i:",i)
#     # model.set_weights(a)
#     # fitness_offsprings = model.evaluate(train_X, train_y, batch_size=128)  # 突变交叉之后的适应度值
#     offsprings = Boundary_control(offsprings)
    print("offsprings:", offsprings)
    indices = fitness_offsprings - fitness_pop  # fitness_pop是P的适应度值，fitness_offsprings是突变交叉之后种群的适应度值,这是适应度值差
    print("indices.shape:",indices.shape)
    # for i in range(pop_number):
    #     if indices[i][0]
    index = [i for i in range(pop_number) if indices[i][0] < 0]  # f(T)<f(pop)的下标
    print("indices:",indices)
    print("index:", index)
    fitness_pop[index] = fitness_offsprings[index]  # 适应度值交换：把原种群中适应度值大的换成小的————产生新的适应度值
    pop[index, :] = offsprings[index, :]  # 种群交换————产生新的种群
    globalminimum[epk] = min(fitness_pop)  # 存放每一轮迭代的最小适应度值
    globalminimizer = pop[index, :]  # 存放T<pop的种群
    print("globalminimizer.shape:", globalminimizer.shape)
    fMin=np.min(fitness_pop)
    bestI = np.argmin(fitness_pop)  # 全局最优适应度值的下标bestI
  #  bestII=np.argmin(globalminimum)
    bestX=pop[bestI, :]#最优解
    #bestX = pop[bestI, :].copy()  ##通过bestI，再找到全局最优个体位置
print("globalminimum.shape:",globalminimum.shape)
print("globalminimum:",globalminimum)
print("globalminimizer.shape:",globalminimizer.shape)
print("globalminimizer:",globalminimizer)
print("bestI:",bestI)
print("bestX:",bestX)


a=reshapeX(weight,bestX)#将最优位置（weight值）转换成矩阵格式，带入模型中
model.set_weights(a)
aa=model.evaluate(train_X,train_y,batch_size=32)
history = model.fit(train_X,train_y,epochs=500, batch_size=32, validation_split=0.2, verbose=2, shuffle=False)
model.save('./result/my_net.model(r)')
#调用模型
#from keras.models import load_model
#model = load_model('my_net.model')
weight=np.array(model.get_weights())
print("weight:",weight)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
#save the result
plt.savefig('.\picture2\wloss(r).png')
pyplot.show()


# make a prediction (traning set)
train_pred = model.predict(train_X)
train_X=train_X[:,-1,:]
print("test_pred:",train_pred.shape)
print("test_data:",train_X.shape)
trainPredict = concatenate((train_pred, train_X[:, 1:]), axis=1)#拼接函数，axis=1表示对应的行列拼接
print("testPredict1:",trainPredict.shape)
trainPredict= scaler.inverse_transform(trainPredict)
trainPredict = trainPredict[:,0:1]
# invert scaling for actual
train_y = train_y.reshape((len(train_y), 1))
train_y = concatenate((train_y, train_X[:, 1:]), axis=1)
train_y = scaler.inverse_transform(train_y)
train_y = train_y[:,0:1]

plt.figure()
plt.plot(train_y[:31069],'b-',label='Raw data',markersize='5',markevery=3)
plt.plot(trainPredict[:31069],'g-',label='ALN_BSA-LSTM(r)',markersize='6',markevery=3)
plt.title('ALN_BSA-LSTM(r)')
plt.xlabel('Data number',fontsize=12)
plt.ylabel('Remainder(mm)',fontsize=12)
plt.legend(['Raw data','ALN_BSA-LSTM(r)'],loc='upper right')
pyplot.savefig('./picture2/walnBSA-train(r).png')#pyplot.savefig('./IBAS-LSTM.png')
plt.show()

# make a prediction (test set)
test_pred = model.predict(test_X)
test_X=test_X[:,-1,:]
print("test_pred:",test_pred.shape)
print("test_data:",test_X.shape)
testPredict = concatenate((test_pred, test_X[:, 1:]), axis=1)#拼接函数，axis=1表示对应的行列拼接
print("testPredict1:",testPredict.shape)
testPredict= scaler.inverse_transform(testPredict)
testPredict = testPredict[:,0:1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
test_y = concatenate((test_y, test_X[:, 1:]), axis=1)
test_y = scaler.inverse_transform(test_y)
test_y = test_y[:,0:1]
print("test_pred:",test_pred.shape)
print("test_X:",test_X.shape)
print("testPredict1:",testPredict.shape)
print("test_y:",test_y.shape)


plt.figure()
plt.plot(globalminimum)  # 最优适应度值
plt.title('Fitness curve')
plt.xlabel('The number of iterations',fontsize=12)
plt.ylabel('The fitness value(reamider)',fontsize=12)
plt.savefig('./picture/walnbsa-dfitness(r).png')
plt.show()

plt.figure()
plt.plot(test_y,'b-',label='Raw data',markersize='5',markevery=3)
plt.plot(testPredict,'g-',label='ALN_BSA-LSTM(r)',markersize='6',markevery=3)
plt.title('ALN_BSA-LSTM(r)')
plt.xlabel('Data number',fontsize=12)
plt.ylabel('Remainder(mm)',fontsize=12)
plt.legend(['Raw data','ALN_BSA-LSTM(r)'],loc='upper right')
pyplot.savefig('./picture2/walnBSA-LSTM2(r).png')#pyplot.savefig('./IBAS-LSTM.png')
plt.show()


# savemat(r'.\result\walnBSA-fitness2(r).mat',{'true':test_y,'pred':testPredict})#savemat('./IBAS-LSTM.mat',{'true':test_y,'pred':testPredict})
test_mape = np.mean(np.abs((testPredict - test_y) / test_y))  # 平均绝对百分比误差
# rmse
test_rmse = np.sqrt(np.mean(np.square(testPredict - test_y)))  # 均方根误差
# mae
test_mae = np.mean(np.abs(testPredict - test_y))  # 平均绝对误差
# R2
test_r2 = r2_score(test_y, testPredict)

mse=mean_squared_error(test_y, testPredict)

print('LSTM测试集的mape:', test_mape, ' rmse:', test_rmse, ' mae:', test_mae, ' R2:', test_r2,'mse:',mse)

data=pd.DataFrame(testPredict)
writer=pd.ExcelWriter(r'./result/walnBSA-LSTM2(r).xls')
data.to_excel(writer,'page_1',float_format='%.5f')
writer.save()
writer.close()