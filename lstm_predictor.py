"""
LSTM趋势预测模块

基于lifePredict_test.py，提供LSTM模型的训练和预测功能：
- 读取CSV数据
- 8/2数据集划分
- 训练LSTM模型（AC准确率>90%或迭代次数不超过100次停止）
- 向后预测20个数据点
- 保存预测结果和组合数据
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import (
    setup_logging,
    calculate_ac,
    create_sequences,
    load_csv_data,
    preprocess_data,
    save_forecast_results
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class LSTMModel(nn.Module):
    """LSTM趋势预测模型"""

    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train_lstm_model(preprocessed_data: dict,
                     max_epochs: int = 100,
                     ac_threshold: float = 0.90,
                     window_size: int = 1,
                     hidden_size: int = 32,
                     num_layers: int = 2,
                     dropout: float = 0.3,
                     batch_size: int = 16,
                     learning_rate: float = 0.001,
                     logger=None) -> dict:
    """
    使用LSTM进行训练，早停条件：AC >= ac_threshold 或 迭代达到 max_epochs

    参数:
        preprocessed_data: 预处理后的数据字典
        max_epochs: 最大迭代次数
        ac_threshold: AC早停阈值
        window_size: 时间窗口大小
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: Dropout率
        batch_size: 批大小
        learning_rate: 学习率
        logger: 日志记录器

    返回:
        包含训练结果的字典
    """
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    X_test = preprocessed_data["X_test"]
    y_test_raw = preprocessed_data["y_test_raw"]
    scaler_y = preprocessed_data["scaler_y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"使用设备: {device}")

    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = LSTMModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_model_state = None
    best_ac = -np.inf

    for epoch in range(1, max_epochs + 1):
        # 训练阶段
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 评估阶段
        model.eval()
        with torch.no_grad():
            X_test_device = X_test_tensor.to(device)
            y_test_pred_scaled = model(X_test_device).cpu().numpy().flatten()

        # 逆变换到原尺度
        y_test_pred_raw = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

        # 计算AC
        test_ac = calculate_ac(y_true=y_test_raw, y_pred=y_test_pred_raw)
        history.append(test_ac)

        if logger:
            logger.info(f"Epoch {epoch}: 测试集AC={test_ac * 100:.2f}%")

        # 记录最佳
        if test_ac > best_ac:
            best_ac = test_ac
            best_model_state = model.state_dict().copy()

        # 早停条件
        if test_ac >= ac_threshold:
            if logger:
                logger.info(f"满足早停条件：测试集AC达到 {test_ac * 100:.2f}% >= {ac_threshold * 100:.2f}% ，停止训练。")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "final_ac": best_ac,
        "history": history,
        "device": device
    }


def forecast_lstm(model: nn.Module,
                  last_series: np.ndarray,
                  scaler_X,
                  scaler_y,
                  window_size: int,
                  steps_ahead: int = 20,
                  device=None) -> np.ndarray:
    """
    基于训练好的LSTM模型，迭代式地向前滚动预测 steps_ahead 个点

    参数:
        model: 训练好的LSTM模型
        last_series: 用于预测的原始序列
        scaler_X: X数据的标准化器
        scaler_y: y数据的标准化器
        window_size: 时间窗口大小
        steps_ahead: 预测步数
        device: 计算设备

    返回:
        预测值数组
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series = last_series.copy().astype(float)

    if len(series) < window_size:
        raise ValueError("用于预测的序列长度不足 window_size。")

    model.eval()
    preds = []

    with torch.no_grad():
        for _ in range(steps_ahead):
            window = series[-window_size:]
            X_input_scaled = scaler_X.transform(window.reshape(-1, 1)).flatten()
            X_tensor = torch.FloatTensor(X_input_scaled).unsqueeze(0).unsqueeze(-1).to(device)
            y_pred_scaled = model(X_tensor).cpu().numpy().flatten()[0]
            y_pred_raw = scaler_y.inverse_transform(np.array([[y_pred_scaled]]))[0, 0]
            preds.append(y_pred_raw)
            series = np.append(series, y_pred_raw)

    return np.array(preds)


def run_lstm_prediction(input_csv: str,
                        output_dir: str,
                        ci_column: str = "CI值_ewma",
                        encoding: str = "gb2312",
                        window_size: int = 1,
                        max_epochs: int = 100,
                        ac_threshold: float = 0.90,
                        hidden_size: int = 32,
                        num_layers: int = 2,
                        dropout: float = 0.3,
                        batch_size: int = 16,
                        learning_rate: float = 0.001,
                        forecast_steps: int = 20,
                        future_time_step: float = 0.5) -> dict:
    """
    LSTM趋势预测完整流程

    参数:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        ci_column: CI值列名
        encoding: 文件编码
        window_size: 时间窗口大小
        max_epochs: 最大迭代次数
        ac_threshold: AC早停阈值
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        dropout: Dropout率
        batch_size: 批大小
        learning_rate: 学习率
        forecast_steps: 预测步数
        future_time_step: 预测时间步长

    返回:
        包含预测结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, "lstm_prediction_log.txt")
    start_time = time.time()

    logger.info("=== LSTM趋势预测流程开始 ===")

    try:
        # 1. 读取数据
        df, ci_series = load_csv_data(input_csv, ci_column, encoding, logger)

        # 2. 数据预处理
        logger.info("开始数据预处理（8/2划分）...")
        preprocessed = preprocess_data(ci_series, split_ratio=0.8, window_size=window_size, logger=logger)

        # 3. 训练模型
        logger.info("开始训练LSTM模型（AC早停）...")
        train_result = train_lstm_model(
            preprocessed,
            max_epochs=max_epochs,
            ac_threshold=ac_threshold,
            window_size=window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            logger=logger
        )
        logger.info(f"训练完成，测试集最终AC={train_result['final_ac'] * 100:.2f}%")

        # 4. 预测未来数据点
        logger.info(f"基于训练好的模型预测未来{forecast_steps}个数据点...")
        forecast_values = forecast_lstm(
            model=train_result["model"],
            last_series=ci_series,
            scaler_X=preprocessed["scaler_X"],
            scaler_y=preprocessed["scaler_y"],
            window_size=window_size,
            steps_ahead=forecast_steps,
            device=train_result["device"]
        )

        # 5. 保存结果
        save_result = save_forecast_results(
            output_dir=output_dir,
            forecast_values=forecast_values,
            forecast_steps=forecast_steps,
            ci_series=ci_series,
            df=df,
            ci_column=ci_column,
            future_time_step=future_time_step,
            encoding=encoding,
            logger=logger
        )

        # 6. 保存训练历史
        hist_file = os.path.join(output_dir, "training_ac_history.csv")
        pd.DataFrame({
            "epoch": np.arange(1, len(train_result["history"]) + 1),
            "test_ac": train_result["history"]
        }).to_csv(hist_file, index=False, encoding=encoding)
        logger.info(f"训练AC历史已保存：{hist_file}")

        # 7. 保存模型
        model_file = os.path.join(output_dir, "lstm_model.pth")
        torch.save({
            'model_state_dict': train_result["model"].state_dict(),
            'window_size': window_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }, model_file)
        logger.info(f"LSTM模型已保存：{model_file}")

        elapsed = time.time() - start_time
        logger.info(f"总耗时：{elapsed:.2f} 秒")
        logger.info("=== LSTM趋势预测流程结束 ===")

        return {
            "model": train_result["model"],
            "scaler_X": preprocessed["scaler_X"],
            "scaler_y": preprocessed["scaler_y"],
            "final_ac": train_result["final_ac"],
            "history": train_result["history"],
            "forecast_values": forecast_values,
            "forecast_file": save_result["forecast_file"],
            "combined_file": save_result["combined_file"],
            "elapsed_time": elapsed
        }

    except Exception as e:
        logger.exception("LSTM预测流程发生异常")
        raise


if __name__ == "__main__":
    # 示例用法
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        ci_column = sys.argv[3] if len(sys.argv) > 3 else "CI值_ewma"
    else:
        print("用法: python lstm_predictor.py <input_csv> <output_dir> [ci_column]")
        sys.exit(1)

    result = run_lstm_prediction(
        input_csv=input_file,
        output_dir=output_dir,
        ci_column=ci_column
    )
    print(f"预测完成，AC准确率: {result['final_ac'] * 100:.2f}%")
