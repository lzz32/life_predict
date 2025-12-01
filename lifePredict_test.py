import os
import argparse
import logging
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_ac(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算国标AC指标：AC = mean(exp(-|err| / |y_true|))
    对 y_true 的 0 做微小数处理，避免除零。
    """
    y_true_abs = np.abs(y_true)
    y_true_abs = np.where(y_true_abs == 0, 1e-8, y_true_abs)
    err = np.abs(y_true - y_pred)
    return float(np.mean(np.exp(-err / y_true_abs)))


def create_sequences(data: np.ndarray, labels: np.ndarray, window_size: int):
    """
    将序列数据构造成监督学习样本：
    X[i] = data[i : i + window_size], y[i] = labels[i + window_size]
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)


def train_with_early_stopping(ci_series: np.ndarray,
                              split_ratio: float = 0.8,
                              window_size: int = 1,
                              max_epochs: int = 100,
                              ac_threshold: float = 0.90,
                              kernel: str = "rbf",
                              C: float = 1.0,
                              epsilon: float = 0.1,
                              logger: logging.Logger = None):
    """
    使用SVR进行训练，数据按 8/2 划分为训练/测试。
    - 训练停止条件：在测试集上 AC >= ac_threshold 或 迭代达到 max_epochs。
    - 迭代策略：固定训练/测试划分，重复拟合（可视为超参稳定或随机性的迭代场景）。
      注：SVR是确定性的；这里的“迭代”是重复训练评估的早停框架以满足用户要求。

    返回：
    - svr_model: 训练好的模型（最后一次满足条件或最大迭代）
    - scaler_X, scaler_y: 标准化器（用于后续预测逆变换）
    - test_ac: 在测试集上的最终AC
    - history: 每次迭代的AC记录
    """
    n = len(ci_series)
    split_idx = int(n * split_ratio)
    if split_idx <= window_size + 1 or split_idx >= n - 1:
        raise ValueError("数据长度不足以按指定比例和窗口大小进行划分，请检查CSV数据量或调整参数。")

    # 划分训练/测试原始序列
    train_raw = ci_series[:split_idx]
    test_raw = ci_series[split_idx:]

    # 标准化：X使用过去窗口，y为下一点
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 训练集X、y的原始构造
    # X使用train_raw，y使用train_raw向后平移
    X_train_seq_raw = train_raw.copy()
    y_train_seq_raw = train_raw.copy()

    # 标准化
    X_train_scaled_flat = scaler_X.fit_transform(X_train_seq_raw.reshape(-1, 1)).flatten()
    y_train_scaled_flat = scaler_y.fit_transform(y_train_seq_raw.reshape(-1, 1)).flatten()

    # 构造训练样本（窗口 -> 下一点）
    X_train, y_train = create_sequences(X_train_scaled_flat, y_train_scaled_flat, window_size)

    # 测试集构造：X使用测试的输入窗口，y使用测试的真实值（原尺度）
    X_test_seq_raw = test_raw.copy()
    y_test_seq_raw = test_raw.copy()

    # 注意：X_test要用训练的scaler_X进行变换，y预测后用scaler_y逆变换
    X_test_scaled_flat = scaler_X.transform(X_test_seq_raw.reshape(-1, 1)).flatten()
    # 构造测试样本（X窗口，y为原尺度的下一点）
    X_test, y_test_next_raw = create_sequences(X_test_scaled_flat, y_test_seq_raw, window_size)

    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("基于当前窗口大小无法构造训练或测试样本，请调整窗口大小或确认数据量。")

    history = []
    best_model = None
    best_ac = -np.inf

    # 由于SVR训练是确定性的，此处“迭代”主要用于早停框架，以满足题述要求。
    for epoch in range(1, max_epochs + 1):
        svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr_model.fit(X_train, y_train)

        # 在测试集上预测（标准化尺度），再逆变换到原尺度
        y_test_pred_scaled = svr_model.predict(X_test)
        y_test_pred_raw = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

        # 评估AC（基于原尺度）
        test_ac = calculate_ac(y_true=y_test_next_raw, y_pred=y_test_pred_raw)
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

    # 若未达到阈值，则使用最佳模型
    final_model = best_model if best_model is not None else svr_model
    final_ac = best_ac

    return final_model, scaler_X, scaler_y, final_ac, history, split_idx


def forecast_next_points(model: SVR,
                         last_series: np.ndarray,
                         scaler_X: StandardScaler,
                         scaler_y: StandardScaler,
                         window_size: int,
                         steps_ahead: int = 20) -> np.ndarray:
    """
    基于训练好的模型，迭代式地向前滚动预测 steps_ahead 个点。
    - 使用最后的原始序列 last_series（原尺度），每次用最近 window_size 个点作为输入，
      经 scaler_X 标准化后进行预测，预测值在标准化尺度上，再用 scaler_y 逆变换到原尺度。
    - 预测点被追加到序列末尾，继续下一步预测。
    """
    series = last_series.copy().astype(float)

    # 如果数据不够窗口大小，无法预测
    if len(series) < window_size:
        raise ValueError("用于预测的序列长度不足 window_size。")

    preds = []
    for _ in range(steps_ahead):
        # 取最近窗口作为输入X
        window = series[-window_size:]
        X_input_scaled = scaler_X.transform(window.reshape(-1, 1)).flatten().reshape(1, -1)

        # 预测（标准化尺度 -> 原尺度）
        y_pred_scaled = model.predict(X_input_scaled)
        y_pred_raw = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).flatten()[0]

        preds.append(y_pred_raw)
        series = np.append(series, y_pred_raw)

    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description="SVR基于CI值进行训练与20步前瞻预测（8/2划分与AC早停）")
    parser.add_argument("--input_csv", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录，用于日志与结果保存")
    parser.add_argument("--ci_column", type=str, default="CI值_ewma", help="CI值列名，若CSV中没有该列，则指定一个替代列名")
    parser.add_argument("--encoding", type=str, default="gb2312", help="CSV文件编码，默认gb2312")
    parser.add_argument("--window_size", type=int, default=1, help="时间窗口大小（默认1，即单步预测）")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练迭代次数（早停上限）")
    parser.add_argument("--ac_threshold", type=float, default=0.90, help="AC早停阈值（默认0.90）")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVR核函数（默认rbf）")
    parser.add_argument("--C", type=float, default=1.0, help="SVR正则化参数C（默认1.0）")
    parser.add_argument("--epsilon", type=float, default=0.1, help="SVR epsilon（默认0.1）")
    parser.add_argument("--forecast_steps", type=int, default=20, help="前瞻预测的步数（默认20）")
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    start_time = time.time()

    try:
        logger.info(f"读取数据：{args.input_csv}")
        df = pd.read_csv(args.input_csv, encoding=args.encoding)
        logger.info(f"数据形状：{df.shape}")

        # 准备CI序列
        if args.ci_column in df.columns:
            ci_series = df[args.ci_column].values
        else:
            raise ValueError(f"未找到列：{args.ci_column}，请确认CSV列名或指定存在的列名。")

        # 训练与早停
        logger.info("开始训练（8/2划分，AC早停）...")
        model, scaler_X, scaler_y, final_ac, history, split_idx = train_with_early_stopping(
            ci_series=ci_series,
            split_ratio=0.8,
            window_size=args.window_size,
            max_epochs=args.max_epochs,
            ac_threshold=args.ac_threshold,
            kernel=args.kernel,
            C=args.C,
            epsilon=args.epsilon,
            logger=logger
        )
        logger.info(f"训练完成，测试集最终AC={final_ac * 100:.2f}%")

        # 基于训练好的模型，使用“训练+测试的整个序列结尾”作为预测起点
        # 为了更稳妥，可使用完整序列进行滚动预测（通常更贴近生产），也可仅用训练集末尾。
        base_series = ci_series  # 使用完整序列结尾
        preds_20 = forecast_next_points(
            model=model,
            last_series=base_series,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            window_size=args.window_size,
            steps_ahead=args.forecast_steps
        )

        # 保存预测结果
        result_df = pd.DataFrame({
            "step": np.arange(1, args.forecast_steps + 1),
            "forecast": preds_20
        })
        out_file = os.path.join(args.output_dir, "forecast_20_steps.csv")
        result_df.to_csv(out_file, index=False, encoding="gb2312")
        logger.info(f"20步预测结果已保存：{out_file}")

        # 保存训练历史AC
        hist_file = os.path.join(args.output_dir, "training_ac_history.csv")
        pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "test_ac": history}).to_csv(
            hist_file, index=False, encoding="gb2312"
        )
        logger.info(f"训练AC历史已保存：{hist_file}")

    except Exception as e:
        logger.exception("发生错误")
        raise e
    finally:
        elapsed = time.time() - start_time
        logger.info(f"总耗时：{elapsed:.2f} 秒")



def run_svr_pipeline(input_csv: str,
                     output_dir: str,
                     log_dir: str,
                     ci_column: str = "CI值_ewma",
                     max_epochs: int = 100,
                     ac_threshold: float = 0.90,
                     forecast_steps: int = 20,
                     future_time_step:float=0.5,
                     window_size: int = 1,
                     kernel: str = "rbf",
                     C: float = 1.0,
                     epsilon: float = 0.1,
                     ) :
    """
    对外暴露的主流程函数，可直接在其它脚本中调用。
    返回一个包含训练与预测结果的字典。
    """
    encoding = "gb2312"
    logger = setup_logging(log_dir)
    start_time = time.time()
    logger.info("=== SVR CI值训练与预测流程开始 ===")

    try:
        logger.info(f"读取数据：{input_csv}")
        df = pd.read_csv(input_csv, encoding=encoding)
        logger.info(f"数据形状：{df.shape}")

        if ci_column not in df.columns:
            raise ValueError(f"未找到列：{ci_column}，请确认CSV列名。")

        ci_series = df[ci_column].values

        logger.info("开始训练（8/2划分，AC早停）...")
        model, scaler_X, scaler_y, final_ac, history, split_idx = train_with_early_stopping(
            ci_series=ci_series,
            split_ratio=0.8,
            window_size=window_size,
            max_epochs=max_epochs,
            ac_threshold=ac_threshold,
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            logger=logger
        )
        logger.info(f"训练完成，测试集最终AC={final_ac * 100:.2f}%")

        base_series = ci_series
        preds_future = forecast_next_points(
            model=model,
            last_series=base_series,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            window_size=window_size,
            steps_ahead=forecast_steps
        )

        # 保存预测结果
        result_df = pd.DataFrame({
            "step": np.arange(1, forecast_steps + 1),
            "forecast": preds_future
        })
        forecast_file = os.path.join(output_dir, f"forecast_{forecast_steps}_steps.csv")
        result_df.to_csv(forecast_file, index=False, encoding="gb2312")
        logger.info(f"{forecast_steps}步预测结果已保存：{forecast_file}")

        # 构造组合文件（原始 + 预测）
        time_column = 't/h'
        if time_column in df.columns:
            # 尽量转换为数值（无法转换的保留原值但预测部分需要数值）
            try:
                time_values = pd.to_numeric(df[time_column], errors="coerce")
            except Exception:
                time_values = pd.Series(np.arange(len(df)), name=time_column)
                logger.warning("t/h 列无法解析为数值，已用顺序索引代替。")
        else:
            # 如果没有 t/h 列，则生成一个按行序号的时间，间隔可设为 future_time_step 或 1
            time_values = pd.Series(np.arange(len(df)), name=time_column)
            logger.warning(f"未找到列 {time_column}，已使用行号作为时间。")

        # 获取最后一个时间点（数值型），无法转换的用最后行索引
        if pd.api.types.is_numeric_dtype(time_values):
            last_time = time_values.iloc[-1] if len(time_values) > 0 else 0.0
            if pd.isna(last_time):
                last_time = float(len(time_values) - 1)
        else:
            # 如果不是数值类型，强制用行号
            last_time = float(len(time_values) - 1)
            logger.warning("时间列非数值，预测时间基于行号。")

        future_times = last_time + future_time_step * np.arange(1, forecast_steps + 1)

        original_part = pd.DataFrame({
            time_column: time_values,
            ci_column: ci_series,
            "is_forecast": [0] * len(ci_series)
        })

        forecast_part = pd.DataFrame({
            time_column: future_times,
            ci_column: preds_future,
            "is_forecast": [1] * forecast_steps
        })

        combined_df = pd.concat([original_part, forecast_part], ignore_index=True)
        combined_file = os.path.join(output_dir, "combined_with_forecast.csv")
        combined_df.to_csv(combined_file, index=False, encoding="gb2312")
        logger.info(f"原始+预测组合文件已保存：{combined_file}")

        # 保存训练历史
        hist_file = os.path.join(output_dir, "training_ac_history.csv")
        pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "test_ac": history}).to_csv(
            hist_file, index=False, encoding="gb2312"
        )
        logger.info(f"训练AC历史已保存：{hist_file}")

        elapsed = time.time() - start_time
        logger.info(f"总耗时：{elapsed:.2f} 秒")
        logger.info("=== 流程结束 ===")

        return {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "final_ac": final_ac,
            "history": history,
            "split_index": split_idx,
            "future_forecast": preds_future,
            "forecast_file": forecast_file,
            "history_file": hist_file
        }

    except Exception as e:
        logger.exception("流程发生异常")
        raise

if __name__ == "__main__":
    input_file = r"D:\Python\25117DT\results\寿命退化门限\交流发电机\交流发电机数据集\gaussian_1d\210-主减速器-平飞2-主减左附件组件径向_2_交流发电机基频_phase_analysis_results.csv"
    output_dir = r"D:\Python\25117DT\results\寿命预测\交流发电机\交流发电机数据集\RULSVR"
    log_dir = output_dir  # 日志文件保存在输出目录
    ciname = 'om1'
    run_svr_pipeline(input_file, output_dir, log_dir, ciname)