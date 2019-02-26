# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9']
x = range(len(names))
y1 = [0.0653,0.0654,0.0655,0.0655,0.0656,0.0759,0.0775,0.0867,0.0975]
y2 = [0.3493,0.3386,0.3181,0.3311,0.3441,0.3524,0.3691,0.3783,0.3891]
y3=[]
for temp1,temp2 in zip(y1,y2):
    y3.append(temp1+temp2)


#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
# plt.plot(x, y1, marker='o', mec='r', mfc='w',label='RMSE')
# plt.plot(x, y2, marker='*', ms=10,  label='Cross-Entropy')
plt.plot(x, y3, marker='D', mec='g', mfc='w',label='RMSE+Cross-Entropy')
plt.legend()  # 让图例生效
plt.ylim((0, 0.5))
plt.xticks(x, names, rotation=45)
plt.yticks([0.00,0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.50])
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("任务权重系数λ") #X轴标签
plt.ylabel("RMSE、Cross-Entropy") #Y轴标签
plt.title("缺失率：60%") #标题

plt.show()