import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import time
import logging
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可重现
np.random.seed(42)


class AsymmetricLoss(nn.Module):
    """非对称损失函数 - 对低估（预测值小于真实值）施加更大惩罚"""

    def __init__(self, under_prediction_penalty=2.0, reduction='mean'):
        """
        参数:
        under_prediction_penalty: 低估惩罚系数，当预测值小于真实值时，损失乘以这个系数
        reduction: 损失归约方式，'mean' 或 'sum'
        """
        super(AsymmetricLoss, self).__init__()
        self.under_prediction_penalty = under_prediction_penalty
        self.reduction = reduction

    def forward(self, predictions, targets):
        # 计算误差
        errors = predictions - targets

        # 创建权重矩阵：低估时权重更大
        weights = torch.where(errors < 0,
                              self.under_prediction_penalty,  # 低估时使用更大的权重
                              1.0)  # 高估或准确时使用正常权重

        # 计算加权的MSE损失
        loss = weights * (errors ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_sequences(data, labels, window_size):
    """创建SVR所需的序列数据"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)


def calculate_ac(y_true, y_pred):
    """计算国标AC指标（避免除零）"""
    y_true_nonzero = np.where(y_true == 0, 1e-8, y_true)  # 避免除零
    a = np.abs(y_true - y_pred)
    AC = np.mean(np.exp(-a / y_true_nonzero))
    return AC


def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, "log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_plot_data(data_dict, filename, output_dir):
    """保存画图数据到CSV文件"""
    plot_data_file = os.path.join(output_dir, filename)
    df = pd.DataFrame(data_dict)
    df.to_csv(plot_data_file, index=False, encoding='gb2312')
    logger.info(f"画图数据已保存到: {plot_data_file}")


def get_sigma_index(df, sigma_col):
    """获取sigma索引列的第一个有效数字"""
    if sigma_col in df.columns:
        values = df[sigma_col].dropna()
        if len(values) > 0:
            return int(values.iloc[0])
    return None


def trend_prediction_svr(df, output_dir):
    """CI值_ewma趋势预测SVR模块"""
    logger.info("开始CI值_ewma趋势预测分析")

    # 获取sigma索引
    # 获取sigma索引
    sigma_2 = get_sigma_index(df, '2sigma_index')
    sigma_3 = df[df['label'] == 2].index.values[0]
    sigma_6 = get_sigma_index(df, '6sigma_index')
    sigma_7 = df[df['label'] == 3].index.values[0]

    logger.info(f"Sigma索引点: 2sigma={sigma_2}, 3sigma={sigma_3}, 6sigma={sigma_6}, 7sigma={sigma_7}")

    # 检查索引有效性
    if any(idx is None for idx in [sigma_2, sigma_3, sigma_6, sigma_7]):
        logger.error("错误：未找到完整的sigma索引点")
        return None, None, None, None, None, None, None, None

    if not (sigma_2 < sigma_3 < sigma_6 < sigma_7):
        logger.error("错误：sigma索引点顺序不正确")
        return None, None, None, None, None, None, None, None

    # 参数设置
    window_size = 1
    epochs = 100
    rolling_step = 20

    # 获取CI值_ewma数据
    ci_data = df['CI值_ewma'].values

    # 初始化结果存储
    all_predictions = []
    all_actuals = []
    prediction_intervals = []

    # 初始训练集和测试集划分
    train_start = sigma_3 - rolling_step - 1
    train_end = sigma_3 - 1
    test_start = sigma_3
    test_end = sigma_7

    # 计算实际的数据点数量
    total_test_points = test_end - test_start + 1
    logger.info(f"总测试点数: {total_test_points}, 从 {test_start} 到 {test_end}")

    current_test_start = test_start
    iteration = 0

    while current_test_start <= test_end:  # 修改条件，包含最后一个点
        iteration += 1
        current_test_end = min(current_test_start + rolling_step - 1, test_end)

        # 实际处理的测试点数量
        actual_test_points = current_test_end - current_test_start + 1
        logger.info(
            f"第{iteration}次滚动预测: 测试集[{current_test_start}, {current_test_end}], 点数: {actual_test_points}")

        # 准备训练数据
        if iteration == 1:
            train_indices = list(range(train_start, train_end + 1))
        else:
            train_indices = list(range(train_start, train_end + 1)) + list(range(test_start, current_test_start))

        # 准备当前测试集
        test_indices = list(range(current_test_start, current_test_end + 1))

        # 修正：标签索引计算
        if iteration == 1:
            label_train_indices = list(range(train_start + window_size, train_end + window_size + 1))
        else:
            label_train_indices = list(range(train_start + window_size, train_end + window_size + 1)) + \
                                  list(range(test_start + window_size, current_test_start + window_size))

        label_test_indices = list(range(current_test_start + window_size, current_test_end + window_size + 1))

        # 检查数据长度
        logger.debug(f"训练输入: {len(train_indices)}, 训练标签: {len(label_train_indices)}")
        logger.debug(f"测试输入: {len(test_indices)}, 测试标签: {len(label_test_indices)}")

        if len(train_indices) < window_size or len(test_indices) < 1:
            logger.warning("训练集或测试集数据不足，停止滚动")
            break

        # 提取数据
        X_train_raw = ci_data[train_indices]
        y_train_raw = ci_data[label_train_indices]

        X_test_raw = ci_data[test_indices]
        y_test_raw = ci_data[label_test_indices]

        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, 1)).flatten()
        y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

        X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, 1)).flatten()

        # 创建序列
        X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, window_size)
        X_test, y_test = create_sequences(X_test_scaled, y_test_raw, window_size)

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning("序列数据创建失败，跳过本次滚动")
            current_test_start += rolling_step
            continue

        # 训练SVR模型
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_model.fit(X_train, y_train)

        # 预测
        predictions_scaled = svr_model.predict(X_test)

        # 反标准化预测结果
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        # 存储结果
        valid_test_points = test_indices[window_size:]
        valid_predictions = predictions[:len(valid_test_points)]
        valid_actuals = y_test_raw[window_size:len(valid_test_points) + window_size]

        all_predictions.extend(valid_predictions)
        all_actuals.extend(valid_actuals)
        prediction_intervals.append({
            'start': valid_test_points[0] if len(valid_test_points) > 0 else current_test_start,
            'end': valid_test_points[-1] if len(valid_test_points) > 0 else current_test_start,
            'predictions': valid_predictions.tolist(),
            'actuals': valid_actuals.tolist()
        })

        # 更新测试起始点
        current_test_start += rolling_step

    # 计算整体AC值
    trend_ac = None
    if len(all_actuals) > 0 and len(all_predictions) > 0:
        trend_ac = calculate_ac(np.array(all_actuals), np.array(all_predictions))
        logger.info(f"寿命预测准确率AC值: {trend_ac * 100:.4f}%")

        # 保存趋势预测结果
        trend_results = {
            '数据索引': list(range(len(all_actuals))),
            '实际CI值': all_actuals,
            '预测CI值': all_predictions,
            '预测误差': np.array(all_actuals) - np.array(all_predictions)
        }

        trend_file = os.path.join(output_dir, "CI值趋势预测结果.csv")
        pd.DataFrame(trend_results).to_csv(trend_file, index=False, encoding='gb2312')
        logger.info(f"趋势预测结果已保存到: {trend_file}")

        return ci_data, sigma_2, sigma_3, sigma_6, sigma_7, all_actuals, all_predictions, trend_ac
    else:
        logger.error("趋势预测未生成有效结果")
        return None, None, None, None, None, None, None, None


def RULSVR(input_file, output_dir, log_dir, ciname):
    # 记录开始时间
    start_time = time.time()

    # 文件路径

    # input_file = r"C:\Users\_taylor\Desktop\LYS_smooth指标数据集\单特征门限划分\输出结果\210-主减速器-平飞2-主减左附件组件径向_2_交流发电机基频_smooth_phase_analysis_results.csv"
    # output_dir = r"C:\Users\_taylor\Desktop\LYS_smooth指标数据集\寿命预测结果文件夹"
    # input_file = r"D:\code_python\Project\25117DT\results\寿命退化门限\交流发电机\gaussian_1d\210-主减速器-平飞2-主减左附件组件径向_2_交流发电机基频_phase_analysis_results - 副本.csv"
    # output_dir = r"D:\code_python\Project\25117DT\results\健康评估\交流发电机\RULSVR"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    global logger
    logger = setup_logging(log_dir)

    logger.info("开始SVR寿命预测分析")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")

    # SVR参数
    window_size = 5
    kernel = 'rbf'
    C = 1.0
    epsilon = 0.1

    # 记录参数设置
    logger.info("模型参数设置:")
    logger.info(f"时间窗口大小: {window_size}")
    logger.info(f"核函数: {kernel}")
    logger.info(f"正则化参数C: {C}")
    logger.info(f"epsilon: {epsilon}")

    try:
        # 1. 读取数据
        logger.info("正在读取数据...")
        df = pd.read_csv(input_file, encoding='gb2312')
        logger.info(f"原始数据形状: {df.shape}")

        # 检查是否有"t/h"列
        has_time_column = 't/h' in df.columns
        if has_time_column:
            logger.info("检测到't/h'列，将用作横坐标")
            time_data = df['t/h'].values
        else:
            logger.info("未检测到't/h'列，将使用索引作为横坐标")

        # 检查是否有CI列
        has_ci_column = 'CI值_ewma' in df.columns
        if not has_ci_column:
            df['CI值_ewma'] = df[ciname]
            logger.info(f"未检测到'CI值_ewma'列，已使用 {ciname} 作为CI值_ewma")

        # 2. 执行CI值趋势预测
        logger.info("开始执行CI值趋势预测...")
        ci_data, sigma_2, sigma_3, sigma_6, sigma_7, trend_actuals, trend_predictions, trend_ac = trend_prediction_svr(
            df, output_dir)
        if trend_ac is not None:
            logger.info(f"CI值趋势预测完成，最终AC值: {trend_ac:.4f}")
        else:
            logger.warning("CI值趋势预测未成功完成")

        # 3. 筛选label为2的数据
        df_label2 = df[df['label'] == 2].copy()
        logger.info(f"label为2的数据行数: {len(df_label2)}")

        if len(df_label2) < 20:
            logger.error("错误：label为2的数据太少")
            return

        # 4. 重新编码索引：从0开始连续编号
        df_label2 = df_label2.reset_index(drop=True)
        df_label2['重新编码索引'] = np.arange(len(df_label2))

        logger.info("重新编码索引后的前几行数据:")
        logger.info(df_label2[['重新编码索引', 'CI值_ewma']].head().to_string())

        # 如果有t/h列，获取对应的t/h数据
        if has_time_column:
            time_data_label2 = df_label2['t/h'].values
            logger.info(f"t/h数据范围: [{time_data_label2.min():.4f}, {time_data_label2.max():.4f}]")

        # 5. 创建新特征：重新编码索引 × CI值_ewma
        ci_data_label2 = df_label2['CI值_ewma'].values
        reindexed_indices = df_label2['重新编码索引'].values

        # 新特征：索引 × CI值_ewma
        new_feature = reindexed_indices * ci_data_label2

        logger.info(f"特征数据点数: {len(new_feature)}")
        logger.info(f"重新编码索引范围: [{reindexed_indices.min()}, {reindexed_indices.max()}]")
        logger.info(f"CI值_ewma范围: [{ci_data_label2.min():.4f}, {ci_data_label2.max():.4f}]")
        logger.info(f"新特征范围: [{new_feature.min():.4f}, {new_feature.max():.4f}]")

        # 6. 创建从100到0的均匀直线作为标签
        y_data = np.linspace(100, 0, len(new_feature))
        logger.info(f"寿命标签范围: [{y_data.min():.2f}, {y_data.max():.2f}]")

        # 7. 数据标准化
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(new_feature.reshape(-1, 1)).flatten()
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

        # 8. 创建序列数据
        X, y = create_sequences(X_scaled, y_scaled, window_size)
        logger.info(f"序列数据形状: X={X.shape}, y={y.shape}")

        # 9. 划分训练集和测试集
        split_idx = int(0.8 * len(X))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"测试集大小: {len(X_test)}")

        # 10. 训练SVR模型
        logger.info("训练SVR模型...")

        # 使用网格搜索优化SVR参数
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['rbf', 'linear']
        }

        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # 使用最佳参数
        best_svr = grid_search.best_estimator_
        logger.info(f"最佳参数: {grid_search.best_params_}")

        # 11. 预测
        # 训练集预测
        y_train_pred_scaled = best_svr.predict(X_train)
        # 测试集预测
        y_test_pred_scaled = best_svr.predict(X_test)
        # 完整数据集预测
        y_full_pred_scaled = best_svr.predict(X)

        # 反标准化预测结果
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        y_full_pred = scaler_y.inverse_transform(y_full_pred_scaled.reshape(-1, 1)).flatten()

        # 反标准化实际值
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_full_actual = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

        # 12. 评估模型
        train_mse = mean_squared_error(y_train_actual, y_train_pred)
        test_mse = mean_squared_error(y_test_actual, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)

        # 计算国标AC指标
        train_ac = calculate_ac(y_train_actual, y_train_pred)
        test_ac = calculate_ac(y_test_actual, y_test_pred)

        # 13. 创建完整预测结果
        reindexed_for_prediction = reindexed_indices[window_size:]
        ci_data_for_prediction = ci_data_label2[window_size:]
        new_feature_for_prediction = new_feature[window_size:]

        # 如果有t/h列，获取对应的t/h数据
        if has_time_column:
            time_data_for_prediction = time_data_label2[window_size:]
            x_coord = time_data_for_prediction
            x_label = 't/h'
        else:
            x_coord = reindexed_for_prediction
            x_label = '重新编码索引'

        results_df = pd.DataFrame({
            x_label: x_coord,
            '重新编码索引': reindexed_for_prediction,
            'CI值_ewma': ci_data_for_prediction,
            '新特征值(索引×CI)': new_feature_for_prediction,
            '实际寿命值': y_full_actual,
            '预测寿命值': y_full_pred,
            '预测误差': y_full_actual - y_full_pred,
            '数据集': ['训练集'] * len(X_train) + ['测试集'] * len(X_test)
        })

        # 14. 保存结果到CSV文件
        output_file = os.path.join(output_dir, "改进_SVR_索引乘CI_寿命预测结果.csv")
        results_df.to_csv(output_file, index=False, encoding='gb2312')
        logger.info(f"\n结果已保存到: {output_file}")

        # 15. 绘制合并的结果图（将两幅图画在一幅图中）
        fig = plt.figure(figsize=(20, 12))

        # 设置橘黄色背景色
        orange_bg_color = 'peachpuff'

        # 第一子图：CI值趋势预测结果
        ax1 = plt.subplot(2, 2, 1)
        if ci_data is not None and sigma_2 is not None:
            # 绘制完整的CI值曲线
            ax1.plot(ci_data, 'b-', alpha=0.7, label='CI值_ewma')

            # 标记sigma点
            sigma_points = [sigma_2, sigma_3, sigma_6, sigma_7]
            sigma_labels = ['2sigma', '3sigma', '6sigma', '7sigma']
            colors = ['red', 'orange', 'green', 'purple']

            for point, label, color in zip(sigma_points, sigma_labels, colors):
                if point < len(ci_data):
                    ax1.axvline(x=point, color=color, linestyle='--', alpha=0.7, label=f'{label} (索引{point})')
                    ax1.plot(point, ci_data[point], 'o', color=color, markersize=8)

            ax1.set_xlabel('时间/h')
            ax1.set_ylabel('CI值_ewma')
            ax1.set_title('CI值_ewma趋势与Sigma划分点')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 第二子图：趋势预测结果
        ax2 = plt.subplot(2, 2, 2)
        if trend_actuals is not None and trend_predictions is not None:
            prediction_indices = list(range(len(trend_actuals)))
            ax2.plot(prediction_indices, trend_actuals, 'b-', label='实际CI值', linewidth=2)
            ax2.plot(prediction_indices, trend_predictions, 'r--', label='预测CI值', linewidth=2)

            ac_text = f' (AC: {trend_ac:.4f})' if trend_ac is not None else ''
            ax2.set_xlabel('相对索引 (从3sigma开始)')
            ax2.set_ylabel('CI值_ewma')
            ax2.set_title(f'CI值趋势预测结果{ac_text}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 第三子图：寿命预测结果
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor(orange_bg_color)

        # 训练集实际值 - 使用折线图
        train_x_coord = x_coord[:len(X_train)]
        test_x_coord = x_coord[len(X_train):]

        ax3.plot(train_x_coord,
                 results_df['实际寿命值'][:len(X_train)],
                 'g-', linewidth=2, label='训练集实际值')
        # 训练集预测值
        ax3.plot(train_x_coord,
                 results_df['预测寿命值'][:len(X_train)],
                 'g--', linewidth=2, label='训练集预测值')
        # 测试集实际值 - 使用折线图
        ax3.plot(test_x_coord,
                 results_df['实际寿命值'][len(X_train):],
                 'r-', linewidth=2, label='测试集实际值')
        # 测试集预测值
        ax3.plot(test_x_coord,
                 results_df['预测寿命值'][len(X_train):],
                 'r--', linewidth=2, label='测试集预测值')

        # 添加训练/测试分界线
        if len(train_x_coord) > 0:
            split_line_x = train_x_coord[-1] if has_time_column else results_df['重新编码索引'][len(X_train)]
            ax3.axvline(x=split_line_x,
                        color='b', linestyle='--', alpha=0.7, label='训练/测试分界线')

        ax3.set_xlabel(x_label)
        ax3.set_ylabel('寿命值')
        ax3.set_title('SVR寿命预测结果')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 第四子图：按Label划分颜色的CI值_ewma
        ax4 = plt.subplot(2, 2, 4)
        # 获取所有不同的label值
        unique_labels = sorted(df['label'].unique())
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']

        # 先绘制所有数据点的CI值，按label分组
        for label_val in unique_labels:
            label_data = df[df['label'] == label_val]
            color = colors[label_val % len(colors)]
            ax4.plot(label_data.index, label_data['CI值_ewma'],
                     color=color, linewidth=2, label=f'label={label_val}')

        # 仅标记label为2的区域（橘黄色背景）
        label2_indices = df[df['label'] == 2].index
        if len(label2_indices) > 0:
            # 找到连续的label=2区域
            label2_ranges = []
            start = label2_indices[0]
            end = label2_indices[0]

            for i in range(1, len(label2_indices)):
                if label2_indices[i] == label2_indices[i - 1] + 1:
                    end = label2_indices[i]
                else:
                    label2_ranges.append((start, end))
                    start = label2_indices[i]
                    end = label2_indices[i]

            label2_ranges.append((start, end))

            # 用橘黄色背景标记label=2的区域
            for start, end in label2_ranges:
                ax4.axvspan(start, end, alpha=0.3, color="orange",
                            label='预测区域' if start == label2_ranges[0][0] else "")

        ax4.set_xlabel('数据索引')
        ax4.set_ylabel('CI值_ewma')
        ax4.set_title('按Label划分颜色的CI值_ewma')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存合并的图片
        combined_plot_file = os.path.join(output_dir, "合并_SVR_寿命预测结果图.png")
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"合并预测结果图已保存到: {combined_plot_file}")

        # 16. 保存模型参数和评估结果
        summary_file = os.path.join(output_dir, "改进_SVR_索引乘CI_模型评估摘要.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("改进的SVR寿命预测模型评估摘要\n")
            f.write("=" * 60 + "\n")
            f.write(f"数据文件: {input_file}\n")
            f.write(f"总数据点数: {len(ci_data_label2)}\n")
            f.write(f"序列数据点数: {len(X)}\n")
            f.write(f"训练集大小: {len(X_train)}\n")
            f.write(f"测试集大小: {len(X_test)}\n")
            f.write(f"时间窗口大小: {window_size}\n")
            f.write(f"核函数: {kernel}\n")
            f.write(f"正则化参数C: {C}\n")
            f.write(f"epsilon: {epsilon}\n")
            f.write(f"最佳参数: {grid_search.best_params_}\n")
            f.write(f"特征: 重新编码索引 × CI值_ewma\n")
            f.write(f"重新编码索引范围: [{reindexed_indices.min()}, {reindexed_indices.max()}]\n")
            f.write(f"CI值范围: [{ci_data_label2.min():.6f}, {ci_data_label2.max():.6f}]\n")
            f.write(f"新特征范围: [{new_feature.min():.6f}, {new_feature.max():.6f}]\n")
            f.write(f"训练集MSE: {train_mse:.6f}\n")
            f.write(f"测试集MSE: {test_mse:.6f}\n")
            f.write(f"训练集RMSE: {train_rmse:.6f}\n")
            f.write(f"测试集RMSE: {test_rmse:.6f}\n")
            f.write(f"训练集R²: {train_r2:.6f}\n")
            f.write(f"测试集R²: {test_r2:.6f}\n")
            f.write(f"训练集AC: {train_ac:.6f}\n")
            f.write(f"测试集AC: {test_ac:.6f}\n")
            if trend_ac is not None:
                f.write(f"CI值趋势预测AC: {trend_ac:.6f}\n")

        logger.info(f"模型摘要已保存到: {summary_file}")

        # 保存完整模型
        import joblib
        model_file = os.path.join(output_dir, "svr_index_ci_complete_model.pkl")
        joblib.dump({
            'model': best_svr,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'window_size': window_size
        }, model_file)
        logger.info(f"完整SVR模型已保存到: {model_file}")

        # 计算并记录总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60

        logger.info(f"分析完成！总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")

        # 保存准确率到csv
        output_data = {'算法模型': 'RULSVR', 'CI值趋势预测AC': trend_ac, '运行时间/s': total_time}
        file_encoding = 'GB2312'
        output_df_path = os.path.join(output_dir, 'training_testing_report.csv')
        output_df = pd.DataFrame([output_data])
        output_df.to_csv(output_df_path, index=False, encoding=file_encoding)
        logger.info(f"准确率数据已保存到: {output_df_path}")

        # plt.show()
        plt.close()

    except FileNotFoundError:
        logger.error(f"错误：找不到文件 {input_file}")
    except KeyError as e:
        logger.error(f"错误：在数据文件中找不到列 {e}")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# 执行分析
if __name__ == "__main__":
    input_file = r"D:\Python\25117DT\results\寿命退化门限\交流发电机\gaussian_1d\210-主减速器-平飞2-主减左附件组件径向_2_交流发电机基频_phase_analysis_results.csv"
    output_dir = r"D:\Python\25117DT\results\健康评估\交流发电机\RULSVR"
    log_dir = output_dir  # 日志文件保存在输出目录
    ciname = 'om1'
    RULSVR(input_file, output_dir, log_dir, ciname)