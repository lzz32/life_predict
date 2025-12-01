# 示例数据和结果 / Example Data and Results

本目录包含示例数据和预测结果，用于演示SVR/CNN/LSTM预测模块的使用。

## 目录结构 / Directory Structure

```
examples/
├── data/
│   └── sample_ci_data.csv       # 示例输入数据
└── results/
    ├── svr/                      # SVR预测结果
    │   ├── forecast_20_steps.csv # 20步预测值
    │   ├── combined_with_forecast.csv  # 原始+预测组合数据
    │   ├── training_ac_history.csv     # 训练AC历史
    │   └── svr_prediction_log.txt      # 日志文件
    ├── cnn/                      # CNN预测结果
    │   └── ...
    └── lstm/                     # LSTM预测结果
        └── ...
```

## 运行示例 / Running Examples

```python
from svr_predictor import run_svr_prediction
from cnn_predictor import run_cnn_prediction
from lstm_predictor import run_lstm_prediction

# SVR预测
result = run_svr_prediction(
    input_csv='examples/data/sample_ci_data.csv',
    output_dir='examples/results/svr',
    ci_column='CI值_ewma',
    forecast_steps=20
)

# CNN预测
result = run_cnn_prediction(
    input_csv='examples/data/sample_ci_data.csv',
    output_dir='examples/results/cnn',
    ci_column='CI值_ewma',
    forecast_steps=20
)

# LSTM预测
result = run_lstm_prediction(
    input_csv='examples/data/sample_ci_data.csv',
    output_dir='examples/results/lstm',
    ci_column='CI值_ewma',
    forecast_steps=20
)
```

## 输出说明 / Output Description

- `forecast_{n}_steps.csv`: 预测的n步数据点
- `combined_with_forecast.csv`: 原始数据与预测数据的组合，`is_forecast`列标识是否为预测值
- `training_ac_history.csv`: 训练过程中每个epoch的AC准确率
- `*_prediction_log.txt`: 详细的运行日志
