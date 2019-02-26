# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['10', '20', '30', '40', '50','60']
x = range(len(names))
y1 = [0.943825354428531,0.8926443933059575, 0.8221485252668691, 0.7725109150511326, 0.7227397709007574, 0.63853132619496]
y2 = [0.942625354428531,0.883625354428531, 0.800625354428531, 0.7717289436338133,0.7117289436338133, 0.6367562283633028]
y3 = [0.937237194872232,0.8722438799733508,0.802129463491012, 0.7517289436338133, 0.7121103558845874,0.63562283633028]
y4 = [0.9271342936779286,0.8825299604531513, 0.792625344609621, 0.7523392019490096, 0.7022057629563412, 0.6254866992789371]
y5 = [0.9172200922960739,0.8828541775742983, 0.7925108463052809, 0.7522819544372568, 0.7020912155574518, 0.6358126350466038]
y6=[0.907014068988063,0.861671741925663,0.7628629943385,0.7208708822021412,0.6808326669994038,0.5863956338359766]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y1, marker='o', mec='r', mfc='w',label='MLT-DAE')
plt.plot(x, y2, marker='*', ms=10,  label='MLT-DAE+LG')
plt.plot(x, y3, marker='D', mec='g', mfc='w',label='DEA+LG')
plt.plot(x, y4, marker='h', ms=10,  label='MIC+LG')
plt.plot(x, y5, marker='H', ms=10,  label='KNN+LG')
plt.plot(x, y6, marker='v', ms=10,  label='MEAN+LG')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
# plt.yticks(np.arange(0.020,0.075, step=0.0005))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("缺失率（%）") #X轴标签
plt.ylabel("ACCURACY") #Y轴标签
plt.title("分类预测效果实验对比结果") #标题
plt.show()