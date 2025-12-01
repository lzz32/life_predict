"""
SVR趋势预测模块

基于lifePredict_test.py，提供SVR模型的训练和预测功能：
- 读取CSV数据
- 8/2数据集划分
- 训练SVR模型（AC准确率>90%或迭代次数不超过100次停止）
- 向后预测20个数据点
- 保存预测结果和组合数据
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVR

from data_preprocessing import (
    setup_logging,
    calculate_ac,
    create_sequences,
    load_csv_data,
    preprocess_data,
    save_forecast_results
)


def train_svr_model(preprocessed_data: dict,
                    max_epochs: int = 100,
                    ac_threshold: float = 0.90,
                    kernel: str = "rbf",
                    C: float = 1.0,
                    epsilon: float = 0.1,
                    logger=None) -> dict:
    """
    使用SVR进行训练，早停条件：AC >= ac_threshold 或 迭代达到 max_epochs

    参数:
        preprocessed_data: 预处理后的数据字典
        max_epochs: 最大迭代次数
        ac_threshold: AC早停阈值
        kernel: SVR核函数
        C: SVR正则化参数
        epsilon: SVR epsilon参数
        logger: 日志记录器

    返回:
        包含训练结果的字典
    """
    X_train = preprocessed_data["X_train"]
    y_train = preprocessed_data["y_train"]
    X_test = preprocessed_data["X_test"]
    y_test_raw = preprocessed_data["y_test_raw"]
    scaler_y = preprocessed_data["scaler_y"]

    history = []
    best_model = None
    best_ac = -np.inf

    # Note: SVR is deterministic - repeated training with same data yields same result.
    # The iteration loop is kept for consistency with the early-stopping framework.
    for epoch in range(1, max_epochs + 1):
        svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr_model.fit(X_train, y_train)

        # 在测试集上预测（标准化尺度），再逆变换到原尺度
        y_test_pred_scaled = svr_model.predict(X_test)
        y_test_pred_raw = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

        # 评估AC（基于原尺度）
        test_ac = calculate_ac(y_true=y_test_raw, y_pred=y_test_pred_raw)
        history.append(test_ac)

        if logger:
            logger.info(f"Epoch {epoch}: 测试集AC={test_ac * 100:.2f}%")

        # 记录最佳
        if test_ac > best_ac:
            best_ac = test_ac
            best_model = svr_model

        # 早停条件
        if test_ac >= ac_threshold:
            if logger:
                logger.info(f"满足早停条件：测试集AC达到 {test_ac * 100:.2f}% >= {ac_threshold * 100:.2f}% ，停止训练。")
            break

    final_model = best_model if best_model is not None else svr_model

    return {
        "model": final_model,
        "final_ac": best_ac,
        "history": history
    }


def forecast_svr(model: SVR,
                 last_series: np.ndarray,
                 scaler_X,
                 scaler_y,
                 window_size: int,
                 steps_ahead: int = 20) -> np.ndarray:
    """
    基于训练好的SVR模型，迭代式地向前滚动预测 steps_ahead 个点

    参数:
        model: 训练好的SVR模型
        last_series: 用于预测的原始序列
        scaler_X: X数据的标准化器
        scaler_y: y数据的标准化器
        window_size: 时间窗口大小
        steps_ahead: 预测步数

    返回:
        预测值数组
    """
    series = last_series.copy().astype(float)

    if len(series) < window_size:
        raise ValueError("用于预测的序列长度不足 window_size。")

    preds = []
    for _ in range(steps_ahead):
        window = series[-window_size:]
        X_input_scaled = scaler_X.transform(window.reshape(-1, 1)).flatten().reshape(1, -1)
        y_pred_scaled = model.predict(X_input_scaled)
        y_pred_raw = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).flatten()[0]
        preds.append(y_pred_raw)
        series = np.append(series, y_pred_raw)

    return np.array(preds)


def run_svr_prediction(input_csv: str,
                       output_dir: str,
                       ci_column: str = "CI值_ewma",
                       encoding: str = "gb2312",
                       window_size: int = 1,
                       max_epochs: int = 100,
                       ac_threshold: float = 0.90,
                       kernel: str = "rbf",
                       C: float = 1.0,
                       epsilon: float = 0.1,
                       forecast_steps: int = 20,
                       future_time_step: float = 0.5) -> dict:
    """
    SVR趋势预测完整流程

    参数:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        ci_column: CI值列名
        encoding: 文件编码
        window_size: 时间窗口大小
        max_epochs: 最大迭代次数
        ac_threshold: AC早停阈值
        kernel: SVR核函数
        C: SVR正则化参数
        epsilon: SVR epsilon参数
        forecast_steps: 预测步数
        future_time_step: 预测时间步长

    返回:
        包含预测结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, "svr_prediction_log.txt")
    start_time = time.time()

    logger.info("=== SVR趋势预测流程开始 ===")

    try:
        # 1. 读取数据
        df, ci_series = load_csv_data(input_csv, ci_column, encoding, logger)

        # 2. 数据预处理
        logger.info("开始数据预处理（8/2划分）...")
        preprocessed = preprocess_data(ci_series, split_ratio=0.8, window_size=window_size, logger=logger)

        # 3. 训练模型
        logger.info("开始训练SVR模型（AC早停）...")
        train_result = train_svr_model(
            preprocessed,
            max_epochs=max_epochs,
            ac_threshold=ac_threshold,
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            logger=logger
        )
        logger.info(f"训练完成，测试集最终AC={train_result['final_ac'] * 100:.2f}%")

        # 4. 预测未来数据点
        logger.info(f"基于训练好的模型预测未来{forecast_steps}个数据点...")
        forecast_values = forecast_svr(
            model=train_result["model"],
            last_series=ci_series,
            scaler_X=preprocessed["scaler_X"],
            scaler_y=preprocessed["scaler_y"],
            window_size=window_size,
            steps_ahead=forecast_steps
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

        elapsed = time.time() - start_time
        logger.info(f"总耗时：{elapsed:.2f} 秒")
        logger.info("=== SVR趋势预测流程结束 ===")

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
        logger.exception("SVR预测流程发生异常")
        raise


if __name__ == "__main__":
    # 示例用法
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
        ci_column = sys.argv[3] if len(sys.argv) > 3 else "CI值_ewma"
    else:
        print("用法: python svr_predictor.py <input_csv> <output_dir> [ci_column]")
        sys.exit(1)

    result = run_svr_prediction(
        input_csv=input_file,
        output_dir=output_dir,
        ci_column=ci_column
    )
    print(f"预测完成，AC准确率: {result['final_ac'] * 100:.2f}%")
