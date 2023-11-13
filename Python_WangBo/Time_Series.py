import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf

data_path = r"C:\Users\wangb\Desktop\time_series.xlsx"
data = pd.read_excel(data_path, index_col=0)
print(data.head)

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
data.plot(title="三年工作量")

# data["diff1"] = data["y"].diff(1).dropna()
# data["diff2"] = data["diff1"].diff(1).dropna()
# data1 = data.loc[:,["y","diff1","diff2"]]
# data1.plot(subplots=True, figsize=(18, 12),title="差分图")

# plot_acf(data).show()
# # plt.show()

# #平稳性检测
# from statsmodels.tsa.stattools import adfuller as ADF
# print("原始序列的ADF检验结果为:", ADF(data['y']))

#差分后的时序图
D_data = data.diff(1).dropna()
D_data.columns = ['y差分']
D_data.plot(title="差分后的时序图") #时序图

#自相关图
plot_acf(D_data).show()


#偏自相关图
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data,lags=4, method="ywm").show() 

# #平稳性检测
# print('差分序列的ADF检验结果为:', ADF(D_data['y差分']))


# #白噪声检验
# from statsmodels.stats.diagnostic import acorr_ljungbox
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))



#差分序列的acf,pacf
1
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(D_data,lags=4,ax=ax1) 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(D_data,lags=4,ax=ax2)
plt.show()


from statsmodels.tsa.arima.model import ARIMA
arima_m = ARIMA(data, order=(1, 1, 1)).fit()
arima_m.summary()
pred = arima_m.predict('2022-10-01', dynamic=True, typ='levels')
print(pred)
# predict_sunspots = arma_mod111.predict('2022-10-1', dynamic=True)