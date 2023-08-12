import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib as mpl
from numpy import polyfit, poly1d
import statistics
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']  # 解决matplotlib无法显示中文的问题
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题

def tongji(error):
    Mean=np.mean(error)
    SD=np.std(error)#标准差
    var=np.var(error)#方差
    Best=np.min(error)
    Worst=np.max(error)
    print("Mean:",Mean,"SD:",SD,"Var:",var,"Best:",Best,"Worst:",Worst)
    return Mean,SD,var,Best,Worst

fitness_sphere=pd.read_excel('./result/sphere.xls', index_col=0)
fitness_schwefel=pd.read_excel('./result/schwefel.xls', index_col=0)
fitness_rastrigins=pd.read_excel('./result/rastrigins.xls', index_col=0)
fitness_Griewanks=pd.read_excel('./result/Griewanks.xls', index_col=0)
fitness_Rosenbrock=pd.read_excel('./result/Rosenbrock.xls', index_col=0)

# print("fitness_sphere.shape:",fitness_sphere.shape)

#1、 shpere
sphereBSA=fitness_sphere.values[:, 0:1]
sphereALN_BSA=fitness_sphere.values[:, 1:2]
sphereGA=fitness_sphere.values[:, 2:3]
spherePSO=fitness_sphere.values[:, 3:4]
sphereZOA=fitness_sphere.values[:, 4:5]
sphereOOA=fitness_sphere.values[:, 5:6]

#2、 schwefel
schwefelPSO=fitness_schwefel.values[:, 0:1]
schwefelGA=fitness_schwefel.values[:, 1:2]
schwefelBSA=fitness_schwefel.values[:, 2:3]
schwefelALN_BSA=fitness_schwefel.values[:, 3:4]
schwefelZOA=fitness_schwefel.values[:, 4:5]
schwefelOOA=fitness_schwefel.values[:, 5:6]

#3、 rastrigins
rastriginsPSO=fitness_rastrigins.values[:, 0:1]
rastriginsGA=fitness_rastrigins.values[:, 1:2]
rastriginsBSA=fitness_rastrigins.values[:, 2:3]
rastriginsALN_BSA=fitness_rastrigins.values[:, 3:4]
rastriginsZOA=fitness_rastrigins.values[:, 4:5]
rastriginsOOA=fitness_rastrigins.values[:, 5:6]

#4、Griewanks
GriewanksPSO=fitness_Griewanks.values[:, 0:1]
GriewanksGA=fitness_Griewanks.values[:, 1:2]
GriewanksBSA=fitness_Griewanks.values[:, 2:3]
GriewanksALN_BSA=fitness_Griewanks.values[:, 3:4]
GriewanksZOA=fitness_Griewanks.values[:, 4:5]
GriewanksOOA=fitness_Griewanks.values[:, 5:6]

#5、Rosenbrock
RosenbrockPSO=fitness_Rosenbrock.values[:, 0:1]
RosenbrockGA=fitness_Rosenbrock.values[:, 1:2]
RosenbrockBSA=fitness_Rosenbrock.values[:, 2:3]
RosenbrockALN_BSA=fitness_Rosenbrock.values[:, 3:4]
RosenbrockZOA=fitness_Rosenbrock.values[:, 4:5]
RosenbrockOOA=fitness_Rosenbrock.values[:, 5:6]


#1、 shpere
print("shpere")
tongji(spherePSO)
tongji(sphereGA)
tongji(sphereZOA)
tongji(sphereOOA)
tongji(sphereBSA)
tongji(sphereALN_BSA)
print("\n")

#2、 schwefel
print("schwefel")
tongji(schwefelPSO)
tongji(schwefelGA)
tongji(schwefelZOA)
tongji(schwefelOOA)
tongji(schwefelBSA)
tongji(schwefelALN_BSA)
print("\n")

#3、 rastrigins
print("rastrigins")
tongji(rastriginsPSO)
tongji(rastriginsGA)
tongji(rastriginsZOA)
tongji(rastriginsOOA)
tongji(rastriginsBSA)
tongji(rastriginsALN_BSA)
print("\n")

#4、Griewanks
print("Griewanks")
tongji(GriewanksPSO)
tongji(GriewanksGA)
tongji(GriewanksZOA)
tongji(GriewanksOOA)
tongji(GriewanksBSA)
tongji(GriewanksALN_BSA)
print("\n")

#5、Rosenbrock
print("Rosenbrock")
tongji(RosenbrockPSO)
tongji(RosenbrockGA)
tongji(RosenbrockZOA)
tongji(RosenbrockOOA)
tongji(RosenbrockBSA)
tongji(RosenbrockALN_BSA)
print("\n")


#1、 shpere
plt.figure(figsize=(8,6))
plt.plot(spherePSO, c='blue', linewidth =3.0, label='PSO')
plt.plot(sphereGA, c='orange', linewidth =3.0, label='GA')
plt.plot(sphereZOA, c='plum', linewidth =3.0, label='ZOA')
plt.plot(sphereOOA, c='lightcoral', linewidth =3.0, label='OOA')
plt.plot(sphereBSA, c='red', linewidth =3.0, label='BSA')
plt.plot(sphereALN_BSA, c='green', linewidth =3.0, label='ALN_BSA')
plt.xlim(xmax=2000,xmin=0)
plt.ylim(ymax=800,ymin=0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("The number of iterations",fontsize=20)
plt.ylabel("Function value",fontsize=20)
plt.legend(loc='upper right', fontsize=17)
plt.savefig('.\picture\sphere(F1).png',dpi=200,bbox_inches = 'tight')
plt.show()

#2、 schwefel
plt.figure(figsize=(8,6))
plt.plot(schwefelPSO, c='blue', linewidth =3.0, label='PSO')
plt.plot(schwefelGA, c='orange', linewidth =3.0, label='GA')
plt.plot(schwefelZOA, c='plum', linewidth =3.0, label='ZOA')
plt.plot(schwefelOOA, c='lightcoral', linewidth =3.0, label='OOA')
plt.plot(schwefelBSA, c='red', linewidth =3.0, label='BSA')
plt.plot(schwefelALN_BSA, c='green', linewidth =3.0, label='ALN_BSA')
plt.xlim(xmax=2000,xmin=0)
plt.ylim(ymax=800,ymin=100)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("The number of iterations",fontsize=20)
plt.ylabel("Function value",fontsize=20)
plt.legend(loc='upper right', fontsize=17)
plt.savefig('.\picture\schwefel(F2).png',dpi=200,bbox_inches = 'tight')
plt.show()

#3、 rastrigins
plt.figure(figsize=(8,6))
plt.plot(rastriginsPSO, c='blue', linewidth =3.0, label='PSO')
plt.plot(rastriginsGA, c='orange', linewidth =3.0, label='GA')
plt.plot(rastriginsZOA, c='plum', linewidth =3.0, label='ZOA')
plt.plot(rastriginsOOA, c='lightcoral', linewidth =3.0, label='OOA')
plt.plot(rastriginsBSA, c='red', linewidth =3.0, label='BSA')
plt.plot(rastriginsALN_BSA, c='green', linewidth =3.0, label='ALN_BSA')
plt.xlim(xmax=2000,xmin=0)
plt.ylim(ymax=400,ymin=50)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("The number of iterations",fontsize=20)
plt.ylabel("Function value",fontsize=20)
plt.legend(loc='upper right', fontsize=17)
plt.savefig('.\picture\wrastrigins(F3).png',dpi=200,bbox_inches = 'tight')
plt.show()

#4、 Griewanks
plt.figure(figsize=(8,6))
plt.plot(GriewanksPSO, c='blue', linewidth =3.0, label='PSO')
plt.plot(GriewanksGA, c='orange', linewidth =3.0, label='GA')
plt.plot(GriewanksZOA, c='plum', linewidth =3.0, label='ZOA')
plt.plot(GriewanksOOA, c='lightcoral', linewidth =3.0, label='OOA')
plt.plot(GriewanksBSA, c='red', linewidth =3.0, label='BSA')
plt.plot(GriewanksALN_BSA, c='green', linewidth =3.0, label='ALN_BSA')
# plt.xlim(xmax=2000,xmin=0)
# plt.ylim(ymax=400,ymin=50)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("The number of iterations",fontsize=20)
plt.ylabel("Function value",fontsize=20)
plt.legend(loc='upper right', fontsize=17)
plt.savefig('.\picture\Griewanks(F4).png',dpi=200,bbox_inches = 'tight')
plt.show()

#5、 Rosenbrock
plt.figure(figsize=(8,6))
plt.plot(RosenbrockPSO, c='blue', linewidth =3.0, label='PSO')
plt.plot(RosenbrockGA, c='orange', linewidth =3.0, label='GA')
plt.plot(RosenbrockZOA, c='plum', linewidth =3.0, label='ZOA')
plt.plot(RosenbrockOOA, c='lightcoral', linewidth =3.0, label='OOA')
plt.plot(RosenbrockBSA, c='red', linewidth =3.0, label='BSA')
plt.plot(RosenbrockALN_BSA, c='green', linewidth =3.0, label='ALN_BSA')
plt.xlim(xmax=2000,xmin=0)
plt.ylim(ymax=400,ymin=0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("The number of iterations",fontsize=20)
plt.ylabel("Function value",fontsize=20)
plt.legend(loc='upper right', fontsize=17)
plt.savefig('.\picture\Rosenbrock(F5).png',dpi=200,bbox_inches = 'tight')
plt.show()
