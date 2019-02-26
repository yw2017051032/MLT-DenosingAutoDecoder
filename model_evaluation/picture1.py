# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9']
x = range(len(names))
y1 = [0.0279,0.0278,0.0279,0.0279,0.0276,0.0277,0.0279,0.0281,0.0282]
y2 = [0.0299,0.0297,0.0198,0.0180,0.0195,0.0254,0.0359,0.0381,0.0324]
y3=[]
for temp1,temp2 in zip(y1,y2):
    y3.append(temp1+temp2)


y4 = [0.0526,0.0525,0.0526,0.0525,0.0526,0.0532,0.0635,0.0638,0.0739]
y5 = [0.2201,0.1997,0.2222,0.2307,0.2325,0.3264,0.3465,0.3681,0.3683]
y6=[]
for temp1,temp2 in zip(y4,y5):
    y6.append(temp1+temp2)

y7 = [0.0653,0.0654,0.0655,0.0655,0.0656,0.0759,0.0775,0.0867,0.0975]
y8 = [0.3493,0.3386,0.3181,0.3311,0.3441,0.3524,0.3691,0.3783,0.3891]
y9=[]
for temp1,temp2 in zip(y7,y8):
    y9.append(temp1+temp2)



#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y3, marker='o',label='缺失率：10%')
plt.plot(x, y6, marker='*',  label='缺失率：40%')
plt.plot(x, y9, marker='D',label='缺失率：60%')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
# plt.yticks(np.arange(0.020,0.075, step=0.0005))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.ylim((0, 0.5))
plt.xlabel("任务权重系数λ") #X轴标签
plt.ylabel("RMSE+Cross-Entropy") #Y轴标签
plt.title("任务权重系数λ对比实验") #标题

plt.show()