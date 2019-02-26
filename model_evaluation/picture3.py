# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['10', '20', '30', '40', '50','60']
x = range(len(names))
y1 = [0.0227,0.0327, 0.0437, 0.0513, 0.0574, 0.0641]
y2 = [0.0249,0.0352, 0.0469, 0.0549, 0.0593, 0.0654]
y3 = [0.0266,0.0372, 0.0460, 0.0521, 0.0586, 0.0666]
y4 = [0.0272,0.0386, 0.0484, 0.0540, 0.0608, 0.0690]
y5 = [0.0291,0.0410, 0.0505, 0.0570, 0.0640, 0.0712]


#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y1, marker='o', mec='r', mfc='w',label='MLT-DAE')
plt.plot(x, y2, marker='*', ms=10,  label='DAE')
plt.plot(x, y3, marker='D', mec='g', mfc='w',label='MIC')
plt.plot(x, y4, marker='h', ms=10,  label='KNN')
plt.plot(x, y5, marker='H', ms=10,  label='MEAN')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.ylim(0, 0.080)  # 限定纵轴的范围
# plt.yticks(np.arange(0.020,0.075, step=0.0005))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("缺失率（%）") #X轴标签
plt.ylabel("RMSE") #Y轴标签
plt.title("缺失值填补实验对比结果") #标题

plt.show()