# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9']
x = range(len(names))
y1 = [0.0279,0.0278,0.0279,0.0279,0.0276,0.0277,0.0279,0.0281,0.0282]
y2 = [0.0199,0.0197,0.0188,0.0180,0.0195,0.0254,0.0259,0.0281,0.0324]
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
# plt.yticks(np.arange(0.020,0.075, step=0.0005))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("任务权重系数λ") #X轴标签
plt.ylabel("RMSE、Cross-Entropy") #Y轴标签
plt.title("缺失率：10%") #标题

plt.show()