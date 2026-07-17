import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
import os
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings('ignore')

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 评估指标和工具
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, confusion_matrix,
                             roc_curve, auc, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 用于诺莫图
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 用于决策曲线
import numpy as np
from sklearn.calibration import calibration_curve

# 用于SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP库未安装，SHAP图功能将不可用。请运行: pip install shap")


class RadiomicsMLPipeline:
    """
    影像组学机器学习流水线
    支持：逻辑回归、支持向量机、梯度提升机、K近邻算法
    """

    def __init__(self, random_state=42):
        """
        初始化机器学习流水线

        参数:
        random_state: 随机种子
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.patient_predictions = {}  # 新增：存储每个病人的预测概率

    def _create_models(self):
        """创建所有模型（不含随机森林）"""
        self.models = {
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ]),

            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', SVC(
                    probability=True,
                    random_state=self.random_state,
                    class_weight='balanced',
                    kernel='rbf',
                    C=1.0
                ))
            ]),

            'GradientBoosting': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    learning_rate=0.1
                ))
            ]),

            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                ))
            ])
        }

    def _clean_numeric_string(self, value):
        """清理数值字符串，处理多个小数点等问题"""
        if pd.isna(value):
            return np.nan

        # 如果是数值型，直接返回
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        # 如果是字符串，进行清理
        if isinstance(value, str):
            # 去除空格
            value = value.strip()

            # 处理多个小数点的问题：保留第一个小数点
            if value.count('.') > 1:
                parts = value.split('.')
                # 保留第一个小数点和最后一个部分
                value = parts[0] + '.' + ''.join(parts[1:])

            # 尝试转换为浮点数
            try:
                return float(value)
            except ValueError:
                # 如果转换失败，尝试其他格式
                # 处理科学计数法
                if 'e' in value.lower():
                    try:
                        return float(value)
                    except:
                        pass

                # 处理百分号
                if '%' in value:
                    try:
                        return float(value.replace('%', '')) / 100
                    except:
                        pass

                # 处理逗号分隔的千分位
                if ',' in value:
                    try:
                        return float(value.replace(',', ''))
                    except:
                        pass

                # 如果所有尝试都失败，返回NaN
                return np.nan

        # 其他类型，尝试转换
        try:
            return float(value)
        except:
            return np.nan

    def _clean_dataframe(self, df):
        """清理数据框中的数值列"""
        df_clean = df.copy()

        # 识别数值列（排除明显的非数值列）
        numeric_cols = []
        for col in df.columns:
            # 跳过ID、文本等列
            if col.lower() in ['id', 'patient_id', 'name', 'diagnosis', 'label', 'target', 'class']:
                continue

            # 检查列是否可以转换为数值
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                # 尝试清理后转换
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    cleaned_values = sample_values.apply(self._clean_numeric_string)
                    if not cleaned_values.isna().all():
                        numeric_cols.append(col)

        print(f"识别到 {len(numeric_cols)} 个数值列")

        # 清理数值列
        for col in numeric_cols:
            df_clean[col] = df[col].apply(self._clean_numeric_string)

        return df_clean

    def _calculate_auc_ci(self, y_true, y_prob, confidence=0.95):
        """
        计算AUC的置信区间（使用自助法）
        """
        n_bootstraps = 1000
        bootstrapped_scores = []

        np.random.seed(self.random_state)
        for i in range(n_bootstraps):
            indices = np.random.randint(0, len(y_prob), len(y_prob))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_prob[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (confidence + (1 - confidence) / 2) * 100

        ci_lower = np.percentile(sorted_scores, lower_percentile)
        ci_upper = np.percentile(sorted_scores, upper_percentile)
        auc_value = roc_auc_score(y_true, y_prob)

        return auc_value, ci_lower, ci_upper

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """
        计算所有评估指标
        """
        metrics = {}

        # 基础指标
        metrics['AUC'] = roc_auc_score(y_true, y_prob)
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Sensitivity'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['Specificity'] = recall_score(y_true, y_pred, pos_label=0)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['F1_Score'] = f1_score(y_true, y_pred)

        # 计算AUC置信区间
        auc_value, ci_lower, ci_upper = self._calculate_auc_ci(y_true, y_prob)
        metrics['AUC'] = auc_value
        metrics['AUC_CI_lower'] = ci_lower
        metrics['AUC_CI_upper'] = ci_upper
        metrics['AUC_CI'] = f"{auc_value:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"

        # 混淆矩阵
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP'] = cm.ravel()
        except:
            metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP'] = 0, 0, 0, 0

        return metrics

    def fit(self, X_train, y_train, feature_names=None):
        """
        训练所有模型
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

        # 创建模型
        self._create_models()

        # 训练每个模型
        for name, model in self.models.items():
            print(f"训练 {name}...")
            model.fit(X_train, y_train)

        print("所有模型训练完成！")

    def predict_all(self, X_train, y_train, X_val, y_val, patient_ids_train=None, patient_ids_val=None):
        """
        为所有病人生成预测概率
        """
        self.patient_predictions = {'Train': {}, 'Validation': {}}

        for name, model in self.models.items():
            print(f"为 {name} 生成预测概率...")

            # 训练集预测
            train_proba = model.predict_proba(X_train)
            self.patient_predictions['Train'][name] = {
                'probabilities': train_proba[:, 1],
                'predictions': model.predict(X_train),
                'true_labels': y_train,
                'patient_ids': patient_ids_train if patient_ids_train is not None else list(range(len(y_train)))
            }

            # 验证集预测
            val_proba = model.predict_proba(X_val)
            self.patient_predictions['Validation'][name] = {
                'probabilities': val_proba[:, 1],
                'predictions': model.predict(X_val),
                'true_labels': y_val,
                'patient_ids': patient_ids_val if patient_ids_val is not None else list(range(len(y_val)))
            }

        return self.patient_predictions

    def evaluate(self, X_train, y_train, X_val, y_val):
        """
        评估所有模型在训练集和验证集上的表现
        """
        self.results = {'Train': {}, 'Validation': {}}

        for name, model in self.models.items():
            print(f"评估 {name}...")

            # 训练集预测
            train_pred = model.predict(X_train)
            train_proba = model.predict_proba(X_train)[:, 1]
            self.results['Train'][name] = self._calculate_metrics(y_train, train_pred, train_proba)

            # 验证集预测
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]
            self.results['Validation'][name] = self._calculate_metrics(y_val, val_pred, val_proba)

        # 整理结果为DataFrame
        results_list = []
        for dataset in ['Train', 'Validation']:
            for model_name, metrics in self.results[dataset].items():
                row = {
                    'Dataset': dataset,
                    'Model': model_name,
                    **metrics
                }
                results_list.append(row)

        results_df = pd.DataFrame(results_list)

        # 重新排列列顺序
        columns_order = ['Dataset', 'Model', 'AUC', 'AUC_CI', 'Accuracy',
                         'Sensitivity', 'Specificity', 'Precision', 'F1_Score',
                         'TP', 'FP', 'FN', 'TN', 'AUC_CI_lower', 'AUC_CI_upper']

        results_df = results_df[columns_order]

        return results_df

    def plot_calibration_curves(self, X_train, y_train, X_val, y_val,
                                n_bins=10, figsize=(14, 6), save_path=None):
        """
        绘制训练集和验证集的校正曲线（并排显示）

        参数:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        n_bins: 分箱数量
        figsize: 图形大小
        save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 颜色列表用于区分不同模型
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))

        # 绘制训练集校正曲线
        ax1.set_title('Calibration Curves - Training Set', fontsize=12, fontweight='bold')
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')

        for idx, (name, model) in enumerate(self.models.items()):
            # 获取训练集预测概率
            y_prob_train = model.predict_proba(X_train)[:, 1]

            # 计算校正曲线
            fraction_pos, mean_pred = calibration_curve(y_train, y_prob_train,
                                                        n_bins=n_bins, strategy='uniform')

            # 绘制校正曲线
            ax1.plot(mean_pred, fraction_pos, 'o-', lw=2, color=colors[idx],
                     label=f'{name}', markersize=6)

            # 计算Brier分数
            brier = brier_score_loss(y_train, y_prob_train)
            ax1.text(0.05, 0.90 - idx * 0.05, f'{name} Brier: {brier:.4f}',
                     transform=ax1.transAxes, fontsize=8, color=colors[idx],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax1.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax1.set_ylabel('Fraction of Positives', fontsize=10)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 绘制验证集校正曲线
        ax2.set_title('Calibration Curves - Validation Set', fontsize=12, fontweight='bold')
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')

        for idx, (name, model) in enumerate(self.models.items()):
            # 获取验证集预测概率
            y_prob_val = model.predict_proba(X_val)[:, 1]

            # 计算校正曲线
            fraction_pos, mean_pred = calibration_curve(y_val, y_prob_val,
                                                        n_bins=n_bins, strategy='uniform')

            # 绘制校正曲线
            ax2.plot(mean_pred, fraction_pos, 'o-', lw=2, color=colors[idx],
                     label=f'{name}', markersize=6)

            # 计算Brier分数
            brier = brier_score_loss(y_val, y_prob_val)
            ax2.text(0.05, 0.90 - idx * 0.05, f'{name} Brier: {brier:.4f}',
                     transform=ax2.transAxes, fontsize=8, color=colors[idx],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax2.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax2.set_ylabel('Fraction of Positives', fontsize=10)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"校正曲线图已保存到: {save_path}")

        plt.show()

    def plot_nomogram(self, X_train, y_train, feature_names=None,
                      figsize=(12, 8), save_path=None, max_features=10):
        """
        为逻辑回归模型绘制诺莫图

        参数:
        X_train, y_train: 训练集特征和标签
        feature_names: 特征名称列表
        figsize: 图形大小
        save_path: 保存路径
        max_features: 最大显示特征数量
        """
        if 'LogisticRegression' not in self.models:
            print("警告: 未找到逻辑回归模型，无法绘制诺莫图")
            return

        # 获取逻辑回归模型
        lr_model = self.models['LogisticRegression']

        # 获取特征名称
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

        # 提取逻辑回归系数
        if hasattr(lr_model, 'named_steps'):
            # 如果是Pipeline
            if 'scaler' in lr_model.named_steps:
                # 如果有标准化，需要调整系数
                scaler = lr_model.named_steps['scaler']
                classifier = lr_model.named_steps['classifier']

                # 获取系数和截距
                coef = classifier.coef_[0]
                intercept = classifier.intercept_[0]

                # 如果有标准化，调整系数
                if hasattr(scaler, 'scale_'):
                    coef = coef / scaler.scale_
                    intercept = intercept - np.sum(coef * scaler.mean_)
            else:
                classifier = lr_model.named_steps['classifier']
                coef = classifier.coef_[0]
                intercept = classifier.intercept_[0]
        else:
            # 如果不是Pipeline
            coef = lr_model.coef_[0]
            intercept = lr_model.intercept_[0]

        # 计算特征重要性（绝对系数）
        feature_importance = np.abs(coef)

        # 选择最重要的特征
        if len(self.feature_names) > max_features:
            top_indices = np.argsort(feature_importance)[-max_features:]
            top_indices = top_indices[::-1]  # 降序排列
        else:
            top_indices = np.arange(len(self.feature_names))
            top_indices = top_indices[np.argsort(feature_importance[top_indices])[::-1]]

        selected_features = [self.feature_names[i] for i in top_indices]
        selected_coef = coef[top_indices]

        # 计算特征值的范围
        feature_ranges = []
        for i in top_indices:
            feature_values = X_train[:, i]
            min_val = np.percentile(feature_values, 5)
            max_val = np.percentile(feature_values, 95)
            feature_ranges.append((min_val, max_val))

        # 创建诺莫图
        fig, ax = plt.subplots(figsize=figsize)

        # 设置y轴位置
        y_positions = np.arange(len(selected_features))

        # 计算每个特征的得分范围
        scores = []
        for i, (coef_val, (min_val, max_val)) in enumerate(zip(selected_coef, feature_ranges)):
            score_min = coef_val * min_val
            score_max = coef_val * max_val
            scores.append((score_min, score_max))

        # 找到全局得分范围
        all_scores = []
        for score_min, score_max in scores:
            all_scores.extend([score_min, score_max])
        all_scores.append(intercept)
        score_min_global = min(all_scores)
        score_max_global = max(all_scores)
        score_range = score_max_global - score_min_global

        # 为每个特征绘制线段
        for i, (feature, coef_val, (score_min, score_max), (min_val, max_val)) in enumerate(
                zip(selected_features, selected_coef, scores, feature_ranges)):
            # 绘制得分线段
            ax.plot([score_min, score_max], [y_positions[i], y_positions[i]],
                    'b-', linewidth=2, alpha=0.7)

            # 标记最小值和最大值
            ax.plot(score_min, y_positions[i], 'bo', markersize=4)
            ax.plot(score_max, y_positions[i], 'bo', markersize=4)

            # 添加特征名称
            ax.text(score_min_global - 0.15 * score_range, y_positions[i],
                    f'{feature}', ha='right', va='center', fontsize=9)

            # 添加特征值标签
            ax.text(score_min, y_positions[i] - 0.2, f'{min_val:.2f}',
                    ha='center', va='top', fontsize=7, color='blue')
            ax.text(score_max, y_positions[i] - 0.2, f'{max_val:.2f}',
                    ha='center', va='top', fontsize=7, color='blue')

        # 添加截距
        ax.plot(intercept, -1, 'ro', markersize=8)
        ax.text(intercept, -1.2, f'Intercept: {intercept:.3f}',
                ha='center', va='top', fontsize=9, color='red')

        # 设置坐标轴
        ax.set_xlim(score_min_global - 0.2 * score_range, score_max_global + 0.2 * score_range)
        ax.set_ylim(-2, len(selected_features))
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Features', fontsize=11)
        ax.set_title('Logistic Regression Nomogram', fontsize=14, fontweight='bold')

        # 添加网格
        ax.grid(True, alpha=0.3, axis='x')

        # 添加总得分轴
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Total Score', fontsize=11)

        # 添加概率轴
        # 计算总得分对应的概率
        def score_to_prob(total_score):
            return 1 / (1 + np.exp(-total_score))

        # 在顶部添加概率刻度
        score_ticks = np.linspace(score_min_global, score_max_global, 11)
        prob_ticks = [score_to_prob(score) for score in score_ticks]
        ax3 = ax.twiny()
        ax3.set_xlim(ax.get_xlim())
        ax3.set_xticks(score_ticks)
        ax3.set_xticklabels([f'{p:.2f}' for p in prob_ticks])
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.spines['top'].set_position(('outward', 30))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"诺莫图已保存到: {save_path}")

        plt.show()

    def plot_decision_curve(self, X_val, y_val,
                            thresholds=np.arange(0, 1, 0.01),
                            figsize=(10, 8), save_path=None):
        """
        绘制决策曲线分析

        参数:
        X_val, y_val: 验证集特征和标签
        thresholds: 阈值范围
        figsize: 图形大小
        save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 颜色列表
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models) + 2))

        # 计算所有病人的净收益
        n = len(y_val)

        # 假设所有病人都不治疗的净收益为0
        # 假设所有病人治疗的净收益
        event_rate = np.sum(y_val) / n
        treat_all = event_rate - thresholds * (1 - event_rate)

        # 绘制参考线
        ax.plot(thresholds, [0] * len(thresholds), 'k--', linewidth=1, label='Treat None')
        ax.plot(thresholds, treat_all, 'k-', linewidth=1, label='Treat All')

        # 为每个模型绘制决策曲线
        for idx, (name, model) in enumerate(self.models.items()):
            # 获取预测概率
            y_prob = model.predict_proba(X_val)[:, 1]

            # 计算不同阈值下的净收益
            net_benefit = []
            for threshold in thresholds:
                # 根据阈值做出决策
                predictions = (y_prob >= threshold).astype(int)

                # 计算真阳性和假阳性
                tp = np.sum((predictions == 1) & (y_val == 1))
                fp = np.sum((predictions == 1) & (y_val == 0))

                # 计算净收益
                if threshold > 0 and threshold < 1:
                    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
                else:
                    nb = 0

                net_benefit.append(nb)

            # 绘制模型曲线
            ax.plot(thresholds, net_benefit, color=colors[idx],
                    linewidth=2, label=name)

        ax.set_xlabel('Threshold Probability', fontsize=12)
        ax.set_ylabel('Net Benefit', fontsize=12)
        ax.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.1, 0.6])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"决策曲线已保存到: {save_path}")

        plt.show()

    def plot_shap_analysis(self, X_train, y_train, X_val, y_val,
                           model_name='LogisticRegression',
                           figsize=(15, 10), save_path=None):
        """
        绘制SHAP分析图

        参数:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        model_name: 要分析的模型名称
        figsize: 图形大小
        save_path: 保存路径
        """
        if not SHAP_AVAILABLE:
            print("SHAP库未安装，无法绘制SHAP图")
            return

        if model_name not in self.models:
            print(f"警告: 未找到模型 {model_name}")
            return

        print(f"正在为 {model_name} 进行SHAP分析...")

        # 获取模型
        model = self.models[model_name]

        # 提取基础模型
        if hasattr(model, 'named_steps'):
            if 'classifier' in model.named_steps:
                base_model = model.named_steps['classifier']
                # 如果有标准化，需要对特征进行预处理
                if 'scaler' in model.named_steps:
                    X_train_shap = model.named_steps['scaler'].transform(X_train)
                    X_val_shap = model.named_steps['scaler'].transform(X_val)
                else:
                    X_train_shap = X_train
                    X_val_shap = X_val
            else:
                base_model = model
                X_train_shap = X_train
                X_val_shap = X_val
        else:
            base_model = model
            X_train_shap = X_train
            X_val_shap = X_val

        # 创建SHAP解释器
        if hasattr(base_model, 'coef_'):
            # 线性模型使用LinearExplainer
            explainer = shap.LinearExplainer(base_model, X_train_shap, feature_perturbation='interventional')
        elif hasattr(base_model, 'feature_importances_'):
            # 树模型使用TreeExplainer
            explainer = shap.TreeExplainer(base_model)
        else:
            # 其他模型使用KernelExplainer
            explainer = shap.KernelExplainer(base_model.predict_proba, X_train_shap[:100])

        # 计算SHAP值
        shap_values = explainer.shap_values(X_val_shap)

        # 如果是二分类，取正类的SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. SHAP概要图
        shap.summary_plot(shap_values, X_val_shap, feature_names=self.feature_names,
                          show=False, max_display=15, plot_size=None)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.sca(axes[0, 0])

        # 2. SHAP条形图（特征重要性）
        shap.summary_plot(shap_values, X_val_shap, feature_names=self.feature_names,
                          plot_type="bar", show=False, max_display=15, plot_size=None)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.sca(axes[0, 1])

        # 3. 瀑布图（第一个样本）
        if len(X_val_shap) > 0:
            shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                 base_values=explainer.expected_value,
                                                 data=X_val_shap[0],
                                                 feature_names=self.feature_names),
                                show=False, max_display=10)
            plt.title(f'SHAP Waterfall Plot (Sample 1) - {model_name}',
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.sca(axes[1, 0])

        # 4. 依赖图（最重要的特征）
        if len(self.feature_names) > 0:
            # 找出最重要的特征
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_feature_idx = np.argmax(feature_importance)
            top_feature = self.feature_names[top_feature_idx]

            shap.dependence_plot(top_feature_idx, shap_values, X_val_shap,
                                 feature_names=self.feature_names,
                                 show=False, alpha=0.5)
            plt.title(f'SHAP Dependence Plot - {top_feature}',
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.sca(axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP分析图已保存到: {save_path}")

        plt.show()

        return shap_values, explainer

    def save_patient_predictions(self, folder_path=None):
        """
        保存每个病人的预测概率到Excel文件

        参数:
        folder_path: 保存文件的文件夹路径，如果为None则弹出文件选择对话框
        """
        if folder_path is None:
            # 使用文件对话框选择保存位置
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory(title="选择保存预测结果的文件夹")

            if not folder_path:  # 用户取消了选择
                print("未选择保存路径，预测结果未保存")
                return

        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 为每个模型分别保存训练集和验证集的预测结果
        for dataset in ['Train', 'Validation']:
            for model_name, pred_data in self.patient_predictions.get(dataset, {}).items():
                # 创建DataFrame
                df = pd.DataFrame({
                    'Patient_ID': pred_data['patient_ids'],
                    'True_Label': pred_data['true_labels'],
                    'Predicted_Label': pred_data['predictions'],
                    'Probability_Class_0': 1 - pred_data['probabilities'],  # 类别0的概率
                    'Probability_Class_1': pred_data['probabilities'],  # 类别1的概率
                    'Prediction_Correct': (pred_data['predictions'] == pred_data['true_labels']).astype(int)
                })

                # 保存为Excel文件
                filename = f"{folder_path}/{model_name}_{dataset}_predictions_{timestamp}.xlsx"
                df.to_excel(filename, index=False)
                print(f"预测结果已保存到: {filename}")

                # 同时保存为CSV文件
                csv_filename = f"{folder_path}/{model_name}_{dataset}_predictions_{timestamp}.csv"
                df.to_csv(csv_filename, index=False)

        # 创建汇总文件（包含所有模型的预测结果）
        all_predictions = []
        for dataset in ['Train', 'Validation']:
            for model_name, pred_data in self.patient_predictions.get(dataset, {}).items():
                for i in range(len(pred_data['patient_ids'])):
                    all_predictions.append({
                        'Dataset': dataset,
                        'Model': model_name,
                        'Patient_ID': pred_data['patient_ids'][i],
                        'True_Label': pred_data['true_labels'][i],
                        'Predicted_Label': pred_data['predictions'][i],
                        'Probability': pred_data['probabilities'][i],
                        'Correct': int(pred_data['predictions'][i] == pred_data['true_labels'][i])
                    })

        if all_predictions:
            summary_df = pd.DataFrame(all_predictions)
            summary_filename = f"{folder_path}/All_Predictions_Summary_{timestamp}.xlsx"

            # 保存到Excel，每个模型一个sheet
            with pd.ExcelWriter(summary_filename, engine='openpyxl') as writer:
                for dataset in ['Train', 'Validation']:
                    for model_name in self.models.keys():
                        model_data = summary_df[(summary_df['Dataset'] == dataset) &
                                                (summary_df['Model'] == model_name)]
                        if not model_data.empty:
                            sheet_name = f"{model_name}_{dataset}"
                            if len(sheet_name) > 31:  # Excel sheet name限制
                                sheet_name = sheet_name[:31]
                            model_data.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"汇总预测结果已保存到: {summary_filename}")

    def save_models(self, folder_path='saved_models'):
        """
        保存训练好的模型

        参数:
        folder_path: 保存模型的文件夹路径
        """
        os.makedirs(folder_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            filename = f"{folder_path}/{name}_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"模型 {name} 已保存到: {filename}")

        # 保存特征名称
        if self.feature_names:
            joblib.dump(self.feature_names, f"{folder_path}/feature_names_{timestamp}.pkl")

    def plot_roc_curves(self, X_train, y_train, X_val, y_val, figsize=(14, 6), save_path=None):
        """
        绘制训练集和验证集的ROC曲线（新增训练集ROC曲线）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 颜色列表用于区分不同模型
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))

        # 绘制训练集ROC曲线
        ax1.set_title('ROC Curves - Training Set', fontsize=12, fontweight='bold')
        for idx, (name, model) in enumerate(self.models.items()):
            # 获取训练集预测概率
            y_prob_train = model.predict_proba(X_train)[:, 1]

            # 计算训练集ROC曲线
            fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
            roc_auc_train = auc(fpr_train, tpr_train)

            # 绘制训练集曲线
            ax1.plot(fpr_train, tpr_train, lw=2, color=colors[idx],
                     label=f'{name} (AUC = {roc_auc_train:.3f})')

        # 绘制对角线
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=10)
        ax1.set_ylabel('True Positive Rate', fontsize=10)
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 绘制验证集ROC曲线
        ax2.set_title('ROC Curves - Validation Set', fontsize=12, fontweight='bold')
        for idx, (name, model) in enumerate(self.models.items()):
            # 获取验证集预测概率
            y_prob_val = model.predict_proba(X_val)[:, 1]

            # 计算验证集ROC曲线
            fpr_val, tpr_val, _ = roc_curve(y_val, y_prob_val)
            roc_auc_val = auc(fpr_val, tpr_val)

            # 绘制验证集曲线
            ax2.plot(fpr_val, tpr_val, lw=2, color=colors[idx],
                     label=f'{name} (AUC = {roc_auc_val:.3f})')

        # 绘制对角线
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=10)
        ax2.set_ylabel('True Positive Rate', fontsize=10)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线图已保存到: {save_path}")

        plt.show()

    def plot_combined_roc_curves(self, X_train, y_train, X_val, y_val, figsize=(10, 8), save_path=None):
        """
        绘制训练集和验证集的ROC曲线在同一张图上
        """
        plt.figure(figsize=figsize)

        # 颜色列表用于区分不同模型
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))

        for idx, (name, model) in enumerate(self.models.items()):
            # 训练集ROC曲线
            y_prob_train = model.predict_proba(X_train)[:, 1]
            fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
            roc_auc_train = auc(fpr_train, tpr_train)

            # 验证集ROC曲线
            y_prob_val = model.predict_proba(X_val)[:, 1]
            fpr_val, tpr_val, _ = roc_curve(y_val, y_prob_val)
            roc_auc_val = auc(fpr_val, tpr_val)

            # 绘制训练集曲线（实线）
            plt.plot(fpr_train, tpr_train, lw=2, color=colors[idx],
                     label=f'{name} (Train AUC = {roc_auc_train:.3f})')

            # 绘制验证集曲线（虚线）
            plt.plot(fpr_val, tpr_val, lw=2, color=colors[idx], linestyle='--',
                     label=f'{name} (Val AUC = {roc_auc_val:.3f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Training vs Validation Sets', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"组合ROC曲线图已保存到: {save_path}")

        plt.show()

    def plot_performance_comparison(self, results_df, figsize=(14, 8), save_path=None):
        """
        绘制模型性能对比图
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score']

        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 获取训练集和验证集数据
            train_data = results_df[results_df['Dataset'] == 'Train']
            val_data = results_df[results_df['Dataset'] == 'Validation']

            models = train_data['Model'].tolist()
            train_values = train_data[metric].tolist()
            val_values = val_data[metric].tolist()

            x = np.arange(len(models))
            width = 0.35

            ax.bar(x - width / 2, train_values, width, label='Train', alpha=0.8, color='skyblue')
            ax.bar(x + width / 2, val_values, width, label='Validation', alpha=0.8, color='lightcoral')

            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")

        plt.show()


def load_existing_files(train_path, val_path, label_column='label', id_column=None):
    """
    加载已有的训练集和验证集文件

    参数:
    train_path: 训练集文件路径
    val_path: 验证集文件路径
    label_column: 标签列名
    id_column: 病人ID列名

    返回:
    X_train, y_train: 训练集特征和标签
    X_val, y_val: 验证集特征和标签
    feature_names: 特征名称列表
    patient_ids_train, patient_ids_val: 病人ID列表
    """
    print("=" * 60)
    print("加载已有训练集和验证集")
    print("=" * 60)

    # 初始化流水线用于数据清理
    pipeline = RadiomicsMLPipeline()

    # 加载训练集
    print(f"加载训练集: {train_path}")
    if train_path.endswith('.csv'):
        train_df = pd.read_csv(train_path)
    elif train_path.endswith(('.xlsx', '.xls')):
        train_df = pd.read_excel(train_path)
    else:
        raise ValueError("只支持CSV和Excel文件格式")

    # 清理训练集数据
    train_clean = pipeline._clean_dataframe(train_df)

    # 加载验证集
    print(f"加载验证集: {val_path}")
    if val_path.endswith('.csv'):
        val_df = pd.read_csv(val_path)
    elif val_path.endswith(('.xlsx', '.xls')):
        val_df = pd.read_excel(val_path)
    else:
        raise ValueError("只支持CSV和Excel文件格式")

    # 清理验证集数据
    val_clean = pipeline._clean_dataframe(val_df)

    print(f"\n训练集原始形状: {train_clean.shape}")
    print(f"验证集原始形状: {val_clean.shape}")

    # 提取病人ID
    patient_ids_train = None
    patient_ids_val = None

    if id_column:
        if id_column in train_clean.columns and id_column in val_clean.columns:
            patient_ids_train = train_clean[id_column].values
            patient_ids_val = val_clean[id_column].values
            train_clean = train_clean.drop(columns=[id_column])
            val_clean = val_clean.drop(columns=[id_column])
            print(f"已提取ID列: {id_column}")
        else:
            print(f"警告: ID列 '{id_column}' 不在两个数据集中")

    # 确认两个数据集的特征列一致
    train_features = [col for col in train_clean.columns if col != label_column]
    val_features = [col for col in val_clean.columns if col != label_column]

    # 找出共同的特征
    common_features = set(train_features) & set(val_features)
    if len(common_features) != len(train_features) or len(common_features) != len(val_features):
        print("警告: 训练集和验证集特征不完全一致")
        print(f"训练集特征数: {len(train_features)}")
        print(f"验证集特征数: {len(val_features)}")
        print(f"共同特征数: {len(common_features)}")

        # 只使用共同特征
        common_features = sorted(list(common_features))
        print(f"\n使用共同特征进行训练和验证")
    else:
        common_features = sorted(train_features)

    # 提取标签
    if label_column in train_clean.columns and label_column in val_clean.columns:
        y_train = train_clean[label_column].values
        y_val = val_clean[label_column].values

        X_train = train_clean[common_features].values
        X_val = val_clean[common_features].values

        feature_names = common_features
    else:
        # 尝试自动找到标签列
        possible_labels = ['label', 'Label', 'target', 'Target', 'y', 'Y', 'class', 'Class']
        label_found = False

        for possible_label in possible_labels:
            if possible_label in train_clean.columns and possible_label in val_clean.columns:
                label_column = possible_label
                y_train = train_clean[label_column].values
                y_val = val_clean[label_column].values

                X_train = train_clean[common_features].values
                X_val = val_clean[common_features].values

                feature_names = common_features
                label_found = True
                print(f"自动识别标签列: {label_column}")
                break

        if not label_found:
            # 如果没有找到标签列，尝试最后一列作为标签
            print("未找到标签列，尝试使用最后一列作为标签")
            y_train = train_clean.iloc[:, -1].values
            y_val = val_clean.iloc[:, -1].values

            X_train = train_clean.iloc[:, :-1].values
            X_val = val_clean.iloc[:, :-1].values

            feature_names = train_clean.columns[:-1].tolist()

    # 确保标签是数值型
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_val = pd.to_numeric(y_val, errors='coerce')

    # 处理缺失值
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    print(f"\n处理后训练集形状: {X_train.shape}")
    print(f"处理后验证集形状: {X_val.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"训练集标签分布: {np.bincount(y_train.astype(int))}")
    print(f"验证集标签分布: {np.bincount(y_val.astype(int))}")

    return X_train, y_train, X_val, y_val, feature_names, patient_ids_train, patient_ids_val


def run_pipeline_with_existing_files(train_path, val_path, label_column='label',
                                     save_results=True, save_models=True,
                                     save_predictions=True, output_dir=None):
    """
    使用已有训练集和验证集文件运行完整的机器学习流水线

    参数:
    train_path: 训练集文件路径
    val_path: 验证集文件路径
    label_column: 标签列名
    save_results: 是否保存结果
    save_models: 是否保存模型
    save_predictions: 是否保存预测概率
    output_dir: 输出目录，如果为None则让用户选择
    """

    print("=" * 60)
    print("影像组学机器学习流水线 - 已有数据集模式")
    print("=" * 60)

    # 如果输出目录未指定，让用户选择
    if output_dir is None:
        root = tk.Tk()
        root.withdraw()
        output_dir = filedialog.askdirectory(title="选择结果保存文件夹")
        if not output_dir:
            print("未选择保存路径，结果将不会被保存")
            save_results = False
            save_models = False
            save_predictions = False
        else:
            # 在选择的文件夹下创建子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, f"radiomics_results_existing_{timestamp}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 询问是否有病人ID列
        print("\n数据集包含病人ID列吗？")
        print("1. 是，我有病人ID列")
        print("2. 否，没有病人ID列")
        id_choice = input("请选择 (1 或 2): ").strip()

        id_column = None
        if id_choice == '1':
            id_column = input("请输入病人ID列的名称: ").strip()
            if not id_column:
                print("未输入ID列名，将不使用ID列")
                id_column = None

        # 加载训练集和验证集
        X_train, y_train, X_val, y_val, feature_names, patient_ids_train, patient_ids_val = load_existing_files(
            train_path, val_path, label_column, id_column=id_column
        )

        # 创建流水线
        pipeline = RadiomicsMLPipeline(random_state=42)

        # 训练模型
        print("\n开始训练模型...")
        pipeline.fit(X_train, y_train, feature_names)

        # 评估模型
        results_df = pipeline.evaluate(X_train, y_train, X_val, y_val)

        # 生成病人预测概率
        pipeline.predict_all(X_train, y_train, X_val, y_val,
                             patient_ids_train, patient_ids_val)

        # 显示结果
        print("\n" + "=" * 60)
        print("模型性能结果:")
        print("=" * 60)
        print(results_df.to_string())

        # 显示部分预测结果示例
        print("\n" + "=" * 60)
        print("预测结果示例 (前5个病人):")
        print("=" * 60)
        for dataset in ['Train', 'Validation']:
            print(f"\n{dataset}集:")
            for model_name in list(pipeline.models.keys())[:2]:  # 只显示前2个模型
                if model_name in pipeline.patient_predictions.get(dataset, {}):
                    pred_data = pipeline.patient_predictions[dataset][model_name]
                    print(f"\n模型: {model_name}")
                    example_df = pd.DataFrame({
                        'Patient_ID': pred_data['patient_ids'][:5],
                        'True_Label': pred_data['true_labels'][:5],
                        'Predicted_Label': pred_data['predictions'][:5],
                        'Probability_Class_1': pred_data['probabilities'][:5]
                    })
                    print(example_df.to_string())

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存结果
        if save_results:
            results_file = os.path.join(output_dir, f'radiomics_results_{timestamp}.xlsx')
            save_results_to_excel(results_df, results_file)

        # 保存模型
        if save_models:
            model_dir = os.path.join(output_dir, 'saved_models')
            pipeline.save_models(model_dir)

        # 保存数据集信息
        save_existing_files_info(train_path, val_path, X_train, y_train, X_val, y_val,
                                 feature_names, output_dir, timestamp,
                                 patient_ids_train, patient_ids_val)

        # 保存预测概率
        if save_predictions:
            print("\n是否要保存病人预测概率？")
            save_pred_choice = input("保存预测概率？(y/n, 默认y): ").strip().lower()
            if save_pred_choice != 'n':
                pipeline.save_patient_predictions(output_dir)

        # 可视化
        print("\n是否生成可视化图表？")
        viz_choice = input("生成可视化图表？(y/n, 默认y): ").strip().lower()
        if viz_choice != 'n':
            # 生成训练集和验证集的ROC曲线图（并排显示）
            roc_path = os.path.join(output_dir, f'roc_curves_both_{timestamp}.png')
            pipeline.plot_roc_curves(X_train, y_train, X_val, y_val, save_path=roc_path)

            # 生成组合ROC曲线图（同一张图）
            combined_roc_path = os.path.join(output_dir, f'roc_curves_combined_{timestamp}.png')
            pipeline.plot_combined_roc_curves(X_train, y_train, X_val, y_val, save_path=combined_roc_path)

            # 生成性能对比图
            perf_path = os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
            pipeline.plot_performance_comparison(results_df, save_path=perf_path)

            # 生成校正曲线图
            cal_path = os.path.join(output_dir, f'calibration_curves_{timestamp}.png')
            pipeline.plot_calibration_curves(X_train, y_train, X_val, y_val, save_path=cal_path)

            # 生成决策曲线图
            dca_path = os.path.join(output_dir, f'decision_curve_{timestamp}.png')
            pipeline.plot_decision_curve(X_val, y_val, save_path=dca_path)

            # 生成诺莫图（基于逻辑回归）
            nomogram_path = os.path.join(output_dir, f'nomogram_{timestamp}.png')
            pipeline.plot_nomogram(X_train, y_train, feature_names, save_path=nomogram_path)

            # 生成SHAP分析图（如果SHAP可用）
            if SHAP_AVAILABLE:
                shap_path = os.path.join(output_dir, f'shap_analysis_{timestamp}.png')
                pipeline.plot_shap_analysis(X_train, y_train, X_val, y_val,
                                            model_name='LogisticRegression', save_path=shap_path)

        return pipeline, results_df

    except Exception as e:
        print(f"流水线运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_existing_files_info(train_path, val_path, X_train, y_train, X_val, y_val,
                             feature_names, output_dir, timestamp,
                             patient_ids_train=None, patient_ids_val=None):
    """保存已有数据集的信息"""
    info_file = os.path.join(output_dir, f'data_info_{timestamp}.txt')

    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("数据集信息\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"训练集文件: {train_path}\n")
        f.write(f"验证集文件: {val_path}\n")
        f.write(f"时间戳: {timestamp}\n\n")

        f.write(f"训练集样本数: {X_train.shape[0]}\n")
        f.write(f"验证集样本数: {X_val.shape[0]}\n")
        f.write(f"特征数量: {len(feature_names)}\n\n")

        f.write("训练集标签分布:\n")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_train, counts_train):
            f.write(f"  标签 {label}: {count} 个样本 ({count / len(y_train) * 100:.1f}%)\n")

        f.write("\n验证集标签分布:\n")
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        for label, count in zip(unique_val, counts_val):
            f.write(f"  标签 {label}: {count} 个样本 ({count / len(y_val) * 100:.1f}%)\n")

        if patient_ids_train is not None:
            f.write(f"\n训练集病人ID示例: {patient_ids_train[:10]}\n")
        if patient_ids_val is not None:
            f.write(f"验证集病人ID示例: {patient_ids_val[:10]}\n")

        f.write("\n特征名称:\n")
        for i, feature in enumerate(feature_names):
            f.write(f"  {i + 1}. {feature}\n")

    print(f"数据集信息已保存到: {info_file}")


def load_and_split_data(file_path, label_column='label', test_size=0.2,
                        split_method='random', random_state=42, id_column=None):
    """
    从单个文件加载数据并划分训练集和验证集

    参数:
    file_path: 数据文件路径
    label_column: 标签列名或索引
    test_size: 验证集比例 (0-1之间)
    split_method: 划分方法 ('random', 'stratified', 'cross_validation')
    random_state: 随机种子
    id_column: 病人ID列名

    返回:
    X_train, y_train: 训练集特征和标签
    X_val, y_val: 验证集特征和标签
    feature_names: 特征名称列表
    patient_ids_train, patient_ids_val: 病人ID列表
    """

    print("=" * 60)
    print("数据加载和划分")
    print("=" * 60)
    print(f"文件: {file_path}")
    print(f"划分方法: {split_method}")
    print(f"验证集比例: {test_size}")

    # 读取文件
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("只支持CSV和Excel文件格式")

    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 清理数据
    pipeline = RadiomicsMLPipeline()
    df_clean = pipeline._clean_dataframe(df)
    print(f"清理后数据形状: {df_clean.shape}")

    # 识别ID列
    patient_ids = None
    if id_column:
        if id_column in df_clean.columns:
            patient_ids = df_clean[id_column].values
            df_clean = df_clean.drop(columns=[id_column])
            print(f"使用ID列: {id_column}")
    else:
        # 自动识别可能的ID列
        possible_id_cols = [col for col in df_clean.columns
                            if col.lower() in ['id', 'patient_id', 'patientid', 'case_id', 'caseid']]
        if possible_id_cols:
            id_column = possible_id_cols[0]
            patient_ids = df_clean[id_column].values
            df_clean = df_clean.drop(columns=[id_column])
            print(f"自动识别ID列: {id_column}")

    # 提取标签和特征
    if isinstance(label_column, str):
        if label_column not in df_clean.columns:
            # 尝试找到标签列
            possible_labels = [col for col in df_clean.columns
                               if col.lower() in ['label', 'target', 'class', 'diagnosis', 'y']]
            if possible_labels:
                label_column = possible_labels[0]
                print(f"自动选择标签列: {label_column}")
            else:
                raise ValueError(f"找不到标签列: {label_column}")

        y = df_clean[label_column].values
        X = df_clean.drop(columns=[label_column]).values
        feature_names = df_clean.drop(columns=[label_column]).columns.tolist()
    else:
        y = df_clean.iloc[:, label_column].values
        X = df_clean.iloc[:, 1:].values if label_column == 0 else df_clean.iloc[:, :label_column].values
        feature_names = df_clean.columns[1:].tolist() if label_column == 0 else df_clean.columns[:label_column].tolist()

    # 确保标签是数值型
    y = pd.to_numeric(y, errors='coerce')

    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)

    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"标签分布: {np.bincount(y.astype(int))}")

    # 根据不同的方法划分数据
    if split_method == 'random':
        # 随机划分
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, shuffle=True
        )

    elif split_method == 'stratified':
        # 分层随机划分（保持类别比例）
        train_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=random_state,
            shuffle=True, stratify=y
        )

    elif split_method == 'cross_validation':
        # 使用交叉验证（返回第一折作为示例）
        skf = StratifiedKFold(n_splits=int(1 / test_size), shuffle=True, random_state=random_state)
        train_idx, val_idx = next(skf.split(X, y))

    else:
        raise ValueError(f"不支持的划分方法: {split_method}")

    # 划分数据
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 划分病人ID
    if patient_ids is not None:
        patient_ids_train = patient_ids[train_idx]
        patient_ids_val = patient_ids[val_idx]
    else:
        patient_ids_train = list(range(len(train_idx)))
        patient_ids_val = list(range(len(val_idx)))

    print(f"\n划分结果:")
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    print(f"训练集标签分布: {np.bincount(y_train.astype(int))}")
    print(f"验证集标签分布: {np.bincount(y_val.astype(int))}")

    return X_train, y_train, X_val, y_val, feature_names, patient_ids_train, patient_ids_val


def run_pipeline_with_split(data_path, label_column='label', test_size=0.2,
                            split_method='stratified', save_results=True,
                            save_models=True, save_predictions=True, output_dir=None):
    """
    使用数据划分运行完整的机器学习流水线

    参数:
    data_path: 数据文件路径
    label_column: 标签列名或索引
    test_size: 验证集比例
    split_method: 划分方法
    save_results: 是否保存结果
    save_models: 是否保存模型
    save_predictions: 是否保存预测概率
    output_dir: 输出目录，如果为None则让用户选择
    """

    print("=" * 60)
    print("影像组学机器学习流水线 - 自动划分模式")
    print("=" * 60)

    # 如果输出目录未指定，让用户选择
    if output_dir is None:
        root = tk.Tk()
        root.withdraw()
        output_dir = filedialog.askdirectory(title="选择结果保存文件夹")
        if not output_dir:
            print("未选择保存路径，结果将不会被保存")
            save_results = False
            save_models = False
            save_predictions = False
        else:
            # 在选择的文件夹下创建子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, f"radiomics_results_{timestamp}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 询问是否有病人ID列
        print("\n数据文件包含病人ID列吗？")
        print("1. 是，我有病人ID列")
        print("2. 否，没有病人ID列")
        id_choice = input("请选择 (1 或 2): ").strip()

        id_column = None
        if id_choice == '1':
            id_column = input("请输入病人ID列的名称: ").strip()
            if not id_column:
                print("未输入ID列名，将使用序号作为ID")
                id_column = None

        # 加载并划分数据
        X_train, y_train, X_val, y_val, feature_names, patient_ids_train, patient_ids_val = load_and_split_data(
            data_path, label_column, test_size, split_method, id_column=id_column
        )

        # 创建流水线
        pipeline = RadiomicsMLPipeline(random_state=42)

        # 训练模型
        pipeline.fit(X_train, y_train, feature_names)

        # 评估模型
        results_df = pipeline.evaluate(X_train, y_train, X_val, y_val)

        # 生成病人预测概率
        pipeline.predict_all(X_train, y_train, X_val, y_val,
                             patient_ids_train, patient_ids_val)

        # 显示结果
        print("\n" + "=" * 60)
        print("模型性能结果:")
        print("=" * 60)
        print(results_df.to_string())

        # 显示部分预测结果示例
        print("\n" + "=" * 60)
        print("预测结果示例 (前5个病人):")
        print("=" * 60)
        for dataset in ['Train', 'Validation']:
            print(f"\n{dataset}集:")
            for model_name in list(pipeline.models.keys())[:2]:  # 只显示前2个模型
                if model_name in pipeline.patient_predictions.get(dataset, {}):
                    pred_data = pipeline.patient_predictions[dataset][model_name]
                    print(f"\n模型: {model_name}")
                    example_df = pd.DataFrame({
                        'Patient_ID': pred_data['patient_ids'][:5],
                        'True_Label': pred_data['true_labels'][:5],
                        'Predicted_Label': pred_data['predictions'][:5],
                        'Probability_Class_1': pred_data['probabilities'][:5]
                    })
                    print(example_df.to_string())

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存结果
        if save_results:
            results_file = os.path.join(output_dir, f'radiomics_results_{timestamp}.xlsx')
            save_results_to_excel(results_df, results_file)

        # 保存模型
        if save_models:
            model_dir = os.path.join(output_dir, 'saved_models')
            pipeline.save_models(model_dir)

        # 保存划分信息
        save_split_info(data_path, X_train, y_train, X_val, y_val,
                        feature_names, test_size, split_method, output_dir, timestamp,
                        patient_ids_train, patient_ids_val)

        # 保存预测概率
        if save_predictions:
            print("\n是否要保存病人预测概率？")
            save_pred_choice = input("保存预测概率？(y/n, 默认y): ").strip().lower()
            if save_pred_choice != 'n':
                pipeline.save_patient_predictions(output_dir)

        # 可视化
        print("\n是否生成可视化图表？")
        viz_choice = input("生成可视化图表？(y/n, 默认y): ").strip().lower()
        if viz_choice != 'n':
            # 生成训练集和验证集的ROC曲线图（并排显示）
            roc_path = os.path.join(output_dir, f'roc_curves_both_{timestamp}.png')
            pipeline.plot_roc_curves(X_train, y_train, X_val, y_val, save_path=roc_path)

            # 生成组合ROC曲线图（同一张图）
            combined_roc_path = os.path.join(output_dir, f'roc_curves_combined_{timestamp}.png')
            pipeline.plot_combined_roc_curves(X_train, y_train, X_val, y_val, save_path=combined_roc_path)

            # 生成性能对比图
            perf_path = os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
            pipeline.plot_performance_comparison(results_df, save_path=perf_path)

            # 生成校正曲线图
            cal_path = os.path.join(output_dir, f'calibration_curves_{timestamp}.png')
            pipeline.plot_calibration_curves(X_train, y_train, X_val, y_val, save_path=cal_path)

            # 生成决策曲线图
            dca_path = os.path.join(output_dir, f'decision_curve_{timestamp}.png')
            pipeline.plot_decision_curve(X_val, y_val, save_path=dca_path)

            # 生成诺莫图（基于逻辑回归）
            nomogram_path = os.path.join(output_dir, f'nomogram_{timestamp}.png')
            pipeline.plot_nomogram(X_train, y_train, feature_names, save_path=nomogram_path)

            # 生成SHAP分析图（如果SHAP可用）
            if SHAP_AVAILABLE:
                shap_path = os.path.join(output_dir, f'shap_analysis_{timestamp}.png')
                pipeline.plot_shap_analysis(X_train, y_train, X_val, y_val,
                                            model_name='LogisticRegression', save_path=shap_path)

        return pipeline, results_df

    except Exception as e:
        print(f"流水线运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_results_to_excel(results_df, filepath):
    """保存结果到Excel文件"""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 保存详细结果
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        # 创建汇总表格
        summary_data = []
        metrics_to_summarize = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_Score']

        for dataset in ['Train', 'Validation']:
            dataset_df = results_df[results_df['Dataset'] == dataset]
            for metric in metrics_to_summarize:
                for model in dataset_df['Model'].unique():
                    value = dataset_df.loc[dataset_df['Model'] == model, metric].values[0]
                    summary_data.append({
                        'Dataset': dataset,
                        'Metric': metric,
                        'Model': model,
                        'Value': value
                    })

        summary_df = pd.DataFrame(summary_data)
        summary_pivot = summary_df.pivot_table(
            index=['Dataset', 'Metric'],
            columns='Model',
            values='Value'
        ).reset_index()

        summary_pivot.to_excel(writer, sheet_name='Summary', index=False)

        # 保存AUC置信区间
        auc_ci_data = []
        for _, row in results_df.iterrows():
            auc_ci_data.append({
                'Dataset': row['Dataset'],
                'Model': row['Model'],
                'AUC_CI': row['AUC_CI']
            })

        auc_ci_df = pd.DataFrame(auc_ci_data)
        auc_ci_df.to_excel(writer, sheet_name='AUC_CI', index=False)

    print(f"结果已保存到: {filepath}")


def save_split_info(data_path, X_train, y_train, X_val, y_val, feature_names,
                    test_size, split_method, output_dir, timestamp,
                    patient_ids_train=None, patient_ids_val=None):
    """保存划分信息"""
    info_file = os.path.join(output_dir, f'split_info_{timestamp}.txt')

    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("数据划分信息\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"原始数据文件: {data_path}\n")
        f.write(f"划分方法: {split_method}\n")
        f.write(f"验证集比例: {test_size}\n")
        f.write(f"时间戳: {timestamp}\n\n")

        f.write(f"训练集样本数: {X_train.shape[0]}\n")
        f.write(f"验证集样本数: {X_val.shape[0]}\n")
        f.write(f"特征数量: {len(feature_names)}\n\n")

        f.write("训练集标签分布:\n")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_train, counts_train):
            f.write(f"  标签 {label}: {count} 个样本 ({count / len(y_train) * 100:.1f}%)\n")

        f.write("\n验证集标签分布:\n")
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        for label, count in zip(unique_val, counts_val):
            f.write(f"  标签 {label}: {count} 个样本 ({count / len(y_val) * 100:.1f}%)\n")

        if patient_ids_train is not None:
            f.write(f"\n训练集病人ID示例: {patient_ids_train[:10]}\n")
        if patient_ids_val is not None:
            f.write(f"验证集病人ID示例: {patient_ids_val[:10]}\n")

        f.write("\n特征名称:\n")
        for i, feature in enumerate(feature_names):
            f.write(f"  {i + 1}. {feature}\n")

    print(f"划分信息已保存到: {info_file}")

    # 保存划分后的数据集
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['label'] = y_train
    if patient_ids_train is not None:
        train_df.insert(0, 'patient_id', patient_ids_train)

    train_file = os.path.join(output_dir, f'train_split_{timestamp}.csv')
    train_df.to_csv(train_file, index=False)

    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['label'] = y_val
    if patient_ids_val is not None:
        val_df.insert(0, 'patient_id', patient_ids_val)

    val_file = os.path.join(output_dir, f'val_split_{timestamp}.csv')
    val_df.to_csv(val_file, index=False)

    print(f"训练集已保存到: {train_file}")
    print(f"验证集已保存到: {val_file}")


def run_cross_validation(data_path, label_column='label', n_splits=5,
                         save_results=True, output_dir='results_cv'):
    """
    运行交叉验证

    参数:
    data_path: 数据文件路径
    label_column: 标签列名
    n_splits: 交叉验证折数
    save_results: 是否保存结果
    output_dir: 输出目录
    """

    print("=" * 60)
    print("交叉验证分析")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    # 清理数据
    pipeline = RadiomicsMLPipeline()
    df_clean = pipeline._clean_dataframe(df)

    # 提取特征和标签
    if isinstance(label_column, str):
        X = df_clean.drop(columns=[label_column]).values
        y = df_clean[label_column].values
        feature_names = df_clean.drop(columns=[label_column]).columns.tolist()
    else:
        X = df_clean.iloc[:, 1:].values
        y = df_clean.iloc[:, label_column].values
        feature_names = df_clean.columns[1:].tolist()

    # 确保标签是数值型
    y = pd.to_numeric(y, errors='coerce')

    # 处理缺失值
    X = np.nan_to_num(X, nan=0.0)

    # 初始化交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每折的结果
    cv_results = []
    fold = 1

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n折 {fold}/{n_splits}:")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")

        # 训练和评估模型
        pipeline = RadiomicsMLPipeline(random_state=42)
        pipeline.fit(X_train, y_train, feature_names)

        for name, model in pipeline.models.items():
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]

            metrics = pipeline._calculate_metrics(y_val, val_pred, val_proba)

            cv_results.append({
                'Fold': fold,
                'Model': name,
                'AUC': metrics['AUC'],
                'Accuracy': metrics['Accuracy'],
                'Sensitivity': metrics['Sensitivity'],
                'Specificity': metrics['Specificity'],
                'F1_Score': metrics['F1_Score']
            })

        fold += 1

    # 汇总结果
    cv_df = pd.DataFrame(cv_results)

    # 计算平均指标
    summary = cv_df.groupby('Model').agg({
        'AUC': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'Sensitivity': ['mean', 'std'],
        'Specificity': ['mean', 'std'],
        'F1_Score': ['mean', 'std']
    }).round(4)

    print("\n" + "=" * 60)
    print("交叉验证结果汇总")
    print("=" * 60)
    print(summary)

    # 保存结果
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'cv_results_{timestamp}.xlsx')

        with pd.ExcelWriter(results_file, engine='openpyxl') as writer:
            cv_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary.to_excel(writer, sheet_name='Summary')

        print(f"\n交叉验证结果已保存到: {results_file}")

    return cv_df, summary


def quick_fix_data_file(file_path, output_path=None):
    """
    快速修复数据文件中的格式问题

    参数:
    file_path: 输入文件路径
    output_path: 输出文件路径（如果为None，则修改原文件）
    """
    if output_path is None:
        output_path = file_path

    print(f"修复文件: {file_path}")

    # 读取文件
    if file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        print("只支持CSV文件修复")
        return

    # 修复多个小数点的问题
    # 正则表达式匹配数字中的多个小数点
    import re

    # 修复模式：1..743 -> 1.743
    fixed_content = re.sub(r'(\d+)\.\.(\d+)', r'\1.\2', content)

    # 修复其他常见问题
    fixed_content = re.sub(r'(\d),(\d)', r'\1\2', fixed_content)  # 移除逗号
    fixed_content = fixed_content.replace('"', '')  # 移除引号
    fixed_content = fixed_content.replace("'", "")  # 移除单引号

    # 写入修复后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"已修复并保存到: {output_path}")


def select_folder_dialog():
    """弹出文件夹选择对话框"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="选择文件夹")
    return folder_path


def select_file_dialog():
    """弹出文件选择对话框"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    return file_path


def main():
    """
    主程序 - 提供多种运行模式
    """
    print("=" * 60)
    print("影像组学机器学习流水线")
    print("=" * 60)

    print("\n请选择运行模式:")
    print("1. 使用已有训练集和验证集文件")
    print("2. 从单个文件自动划分训练集和验证集")
    print("3. 交叉验证分析")
    print("4. 修复数据文件格式")
    print("5. 查看数据文件")

    choice = input("\n请选择模式 (1-5): ").strip()

    if choice == '1':
        # 模式1：使用已有训练集和验证集
        print("\n选择训练集文件...")
        train_path = select_file_dialog()
        if not train_path:
            print("未选择训练集文件")
            return

        print("\n选择验证集文件...")
        val_path = select_file_dialog()
        if not val_path:
            print("未选择验证集文件")
            return

        # 询问标签列名
        print("\n请输入标签列名称（直接回车使用默认值'label'）:")
        label_column = input("标签列名: ").strip()
        if not label_column:
            label_column = 'label'

        # 运行流水线
        pipeline, results = run_pipeline_with_existing_files(
            train_path=train_path,
            val_path=val_path,
            label_column=label_column,
            save_results=True,
            save_models=True,
            save_predictions=True,
            output_dir=None  # 让用户在运行时选择
        )

        if pipeline and results is not None:
            print("\n" + "=" * 60)
            print("流水线运行完成！")
            print("=" * 60)
        else:
            print("\n流水线运行失败！")

    elif choice == '2':
        # 模式2：从单个文件自动划分
        print("\n选择数据文件...")
        data_path = select_file_dialog()
        if not data_path:
            print("未选择数据文件")
            return

        print("\n请选择划分方法:")
        print("1. 随机划分")
        print("2. 分层随机划分（推荐，保持类别比例）")
        print("3. 交叉验证划分")

        split_choice = input("请选择划分方法 (1-3): ").strip()

        if split_choice == '1':
            split_method = 'random'
        elif split_choice == '2':
            split_method = 'stratified'
        elif split_choice == '3':
            split_method = 'cross_validation'
        else:
            print("无效选择，使用默认方法（分层随机划分）")
            split_method = 'stratified'

        # 获取验证集比例
        test_size_input = input("请输入验证集比例 (0.1-0.4，默认0.2): ").strip()
        try:
            test_size = float(test_size_input) if test_size_input else 0.2
            if not 0.1 <= test_size <= 0.4:
                print("比例应在0.1到0.4之间，使用默认值0.2")
                test_size = 0.2
        except:
            print("输入无效，使用默认值0.2")
            test_size = 0.2

        # 运行自动划分流水线
        pipeline, results = run_pipeline_with_split(
            data_path=data_path,
            label_column='label',
            test_size=test_size,
            split_method=split_method,
            save_results=True,
            save_models=True,
            save_predictions=True,
            output_dir=None  # 让用户在运行时选择
        )

    elif choice == '3':
        # 模式3：交叉验证
        print("\n选择数据文件...")
        data_path = select_file_dialog()
        if not data_path:
            print("未选择数据文件")
            return

        # 运行交叉验证
        cv_df, summary = run_cross_validation(
            data_path=data_path,
            label_column='label',
            n_splits=5,
            save_results=True,
            output_dir='results_cv'
        )

    elif choice == '4':
        # 模式4：修复数据文件
        print("\n选择需要修复的文件...")
        file_path = select_file_dialog()
        if file_path:
            output_path = input("请输入输出文件路径 (直接回车覆盖原文件): ").strip()
            if not output_path:
                output_path = None
            quick_fix_data_file(file_path, output_path)
        else:
            print("未选择文件")

    elif choice == '5':
        # 模式5：查看数据文件
        print("\n选择要查看的文件...")
        file_path = select_file_dialog()
        if file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print("不支持的文件格式")
                df = None

            if df is not None:
                print(f"文件形状: {df.shape}")
                print("\n前5行数据:")
                print(df.head())

                print("\n列名:")
                print(df.columns.tolist())

                print("\n数据类型:")
                print(df.dtypes)

                # 检查数值问题
                print("\n检查数值列问题...")
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # 检查是否有多个小数点
                        sample = df[col].dropna().head(20)
                        for val in sample:
                            if isinstance(val, str) and val.count('.') > 1:
                                print(f"列 '{col}' 中发现多个小数点: {val}")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()