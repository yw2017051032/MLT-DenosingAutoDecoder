# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9']
x = range(len(names))
y1 = [0.0526,0.0525,0.0526,0.0525,0.0526,0.0532,0.0635,0.0638,0.0739]
y2 = [0.2201,0.1997,0.2222,0.2307,0.2325,0.3264,0.3465,0.3681,0.4283]
y3=[]
for temp1,temp2 in zip(y1,y2):
    y3.append(temp1+temp2)


#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y1, marker='o', mec='r', mfc='w',label='RMSE')
plt.plot(x, y2, marker='*', ms=10,  label='Cross-Entropy')
plt.plot(x, y3, marker='D', mec='g', mfc='w',label='RMSE+Cross-Entropy')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.ylim((0, 0.5))
plt.yticks([0.00,0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50])
# plt.yticks(np.arange(0.020,0.075, step=0.0005))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("任务权重系数λ") #X轴标签
plt.ylabel("RMSE、Cross-Entropy") #Y轴标签
plt.title("缺失率：40%") #标题

plt.show()