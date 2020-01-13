# Stock-Price-Prediction-using-Hidden-Markov-Model

创新工场金融项目：隐马尔科夫模型对时序数据的预测

## 数据
通过[Python api](http://tushare.org/)获取的沪深三百股指的每日收盘价格，共612条，其中训练集489条（80%），测试集123条（20%）

## 任务
1. 实现一个利用Baum-Welch训练的HMM Python模块，利用该HMM模块对金融股票数据进行预测。
2. 基于该时间序列数据的每日收盘价格，训练4个HMM模型， HMM的隐状态分别是 4 8 16 32
3. 预测收盘价格

## 结果
模型收敛过程

![模型收敛过程](https://github.com/HouchangX-AI/Stock-Price-Prediction-using-Hidden-Markov-Model/blob/master/figs/Figure_1.png)

模型预测结果：(测试集)
<p float="left">
  <img src="https://github.com/HouchangX-AI/Stock-Price-Prediction-using-Hidden-Markov-Model/blob/master/figs/4_hidden_states_predictions.png" width="420" />
  <img src="https://github.com/HouchangX-AI/Stock-Price-Prediction-using-Hidden-Markov-Model/blob/master/figs/8_hidden_states_predictions.png" width="420" /> 
</p>

<p float="left">
  <img src="https://github.com/HouchangX-AI/Stock-Price-Prediction-using-Hidden-Markov-Model/blob/master/figs/16_hidden_states_predictions.png" width="420" />
  <img src="https://github.com/HouchangX-AI/Stock-Price-Prediction-using-Hidden-Markov-Model/blob/master/figs/32_hidden_states_predictions.png" width="420" /> 
</p>

