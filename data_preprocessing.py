"""
统一数据预处理模块

本模块提供用于寿命预测的统一数据预处理功能，包括：
- CSV数据读取
- 数据标准化
- 序列数据创建
- 训练/测试集划分
- AC准确率计算
- 日志设置
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def setup_logging(output_dir: str, log_name: str = "log.txt") -> logging.Logger:
    """
    设置日志记录

    参数:
        output_dir: 日志文件保存目录
        log_name: 日志文件名

    返回:
        logger: 日志记录器
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, log_name)

    # 创建一个新的logger，避免重复配置
    logger = logging.getLogger(f"life_predict_{output_dir}")
    logger.setLevel(logging.INFO)

    # 清除现有的handlers
    logger.handlers = []

    # 添加文件handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def calculate_ac(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算国标AC指标：AC = mean(exp(-|err| / |y_true|))
    对 y_true 的 0 做微小数处理，避免除零。

    参数:
        y_true: 真实值数组
        y_pred: 预测值数组

    返回:
        AC准确率值
    """
    y_true_abs = np.abs(y_true)
    y_true_abs = np.where(y_true_abs == 0, 1e-8, y_true_abs)
    err = np.abs(y_true - y_pred)
    return float(np.mean(np.exp(-err / y_true_abs)))


def create_sequences(data: np.ndarray, labels: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将序列数据构造成监督学习样本：
    X[i] = data[i : i + window_size], y[i] = labels[i + window_size]

    参数:
        data: 输入特征数据
        labels: 标签数据
        window_size: 时间窗口大小

    返回:
        X: 特征数组 (n_samples, window_size)
        y: 标签数组 (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)


def load_csv_data(input_csv: str, ci_column: str = "CI值_ewma", 
                  encoding: str = "gb2312", logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    读取CSV文件并提取CI值序列

    参数:
        input_csv: CSV文件路径
        ci_column: CI值列名
        encoding: 文件编码
        logger: 日志记录器

    返回:
        df: 数据框
        ci_series: CI值序列
    """
    if logger:
        logger.info(f"读取数据：{input_csv}")

    df = pd.read_csv(input_csv, encoding=encoding)

    if logger:
        logger.info(f"数据形状：{df.shape}")

    if ci_column in df.columns:
        ci_series = df[ci_column].values
    else:
        raise ValueError(f"未找到列：{ci_column}，请确认CSV列名或指定存在的列名。")

    return df, ci_series


def preprocess_data(ci_series: np.ndarray, 
                    split_ratio: float = 0.8,
                    window_size: int = 1,
                    logger: Optional[logging.Logger] = None) -> dict:
    """
    统一数据预处理流程：
    1. 按8/2划分训练/测试集
    2. 数据标准化
    3. 创建序列数据

    参数:
        ci_series: CI值序列
        split_ratio: 训练集比例（默认0.8，即8/2划分）
        window_size: 时间窗口大小
        logger: 日志记录器

    返回:
        包含预处理后数据的字典
    """
    n = len(ci_series)
    split_idx = int(n * split_ratio)

    if split_idx <= window_size + 1 or split_idx >= n - 1:
        raise ValueError("数据长度不足以按指定比例和窗口大小进行划分，请检查CSV数据量或调整参数。")

    # 划分训练/测试原始序列
    train_raw = ci_series[:split_idx]
    test_raw = ci_series[split_idx:]

    if logger:
        logger.info(f"训练集大小: {len(train_raw)}, 测试集大小: {len(test_raw)}")

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 训练集标准化
    X_train_scaled_flat = scaler_X.fit_transform(train_raw.reshape(-1, 1)).flatten()
    y_train_scaled_flat = scaler_y.fit_transform(train_raw.reshape(-1, 1)).flatten()

    # 构造训练样本
    X_train, y_train = create_sequences(X_train_scaled_flat, y_train_scaled_flat, window_size)

    # 测试集标准化（使用训练集的scaler）
    X_test_scaled_flat = scaler_X.transform(test_raw.reshape(-1, 1)).flatten()
    # 构造测试样本（y使用原始值以便计算AC）
    X_test, y_test_raw = create_sequences(X_test_scaled_flat, test_raw, window_size)

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("基于当前窗口大小无法构造训练或测试样本，请调整窗口大小或确认数据量。")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test_raw": y_test_raw,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "split_idx": split_idx,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "ci_series": ci_series
    }


def save_forecast_results(output_dir: str,
                          forecast_values: np.ndarray,
                          forecast_steps: int,
                          ci_series: np.ndarray,
                          df: pd.DataFrame,
                          ci_column: str = "CI值_ewma",
                          time_column: str = "t/h",
                          future_time_step: float = 0.5,
                          encoding: str = "gb2312",
                          logger: Optional[logging.Logger] = None) -> dict:
    """
    保存预测结果

    参数:
        output_dir: 输出目录
        forecast_values: 预测值数组
        forecast_steps: 预测步数
        ci_series: 原始CI值序列
        df: 原始数据框
        ci_column: CI值列名
        time_column: 时间列名
        future_time_step: 预测时间步长
        encoding: 文件编码
        logger: 日志记录器

    返回:
        包含保存文件路径的字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测结果
    result_df = pd.DataFrame({
        "step": np.arange(1, forecast_steps + 1),
        "forecast": forecast_values
    })
    forecast_file = os.path.join(output_dir, f"forecast_{forecast_steps}_steps.csv")
    result_df.to_csv(forecast_file, index=False, encoding=encoding)
    if logger:
        logger.info(f"{forecast_steps}步预测结果已保存：{forecast_file}")

    # 构造组合文件（原始 + 预测）
    if time_column in df.columns:
        try:
            time_values = pd.to_numeric(df[time_column], errors="coerce")
        except Exception:
            time_values = pd.Series(np.arange(len(df)), name=time_column)
            if logger:
                logger.warning("t/h 列无法解析为数值，已用顺序索引代替。")
    else:
        time_values = pd.Series(np.arange(len(df)), name=time_column)
        if logger:
            logger.warning(f"未找到列 {time_column}，已使用行号作为时间。")

    # 获取最后一个时间点
    if pd.api.types.is_numeric_dtype(time_values):
        last_time = time_values.iloc[-1] if len(time_values) > 0 else 0.0
        if pd.isna(last_time):
            last_time = float(len(time_values) - 1)
    else:
        last_time = float(len(time_values) - 1)

    future_times = last_time + future_time_step * np.arange(1, forecast_steps + 1)

    original_part = pd.DataFrame({
        time_column: time_values,
        ci_column: ci_series,
        "is_forecast": [0] * len(ci_series)
    })

    forecast_part = pd.DataFrame({
        time_column: future_times,
        ci_column: forecast_values,
        "is_forecast": [1] * forecast_steps
    })

    combined_df = pd.concat([original_part, forecast_part], ignore_index=True)
    combined_file = os.path.join(output_dir, "combined_with_forecast.csv")
    combined_df.to_csv(combined_file, index=False, encoding=encoding)
    if logger:
        logger.info(f"原始+预测组合文件已保存：{combined_file}")

    return {
        "forecast_file": forecast_file,
        "combined_file": combined_file
    }
