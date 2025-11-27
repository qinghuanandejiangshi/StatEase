import pandas as pd
import numpy as np
from scipy import stats

def correlation_analysis(df, var1_col, var2_col):
    """
    执行相关性分析 (Pearson/Spearman)
    :param df: DataFrame
    :param var1_col: 变量1列名
    :param var2_col: 变量2列名
    :return: 格式化的文本报告
    """
    # 1. 数据准备
    data = df[[var1_col, var2_col]].dropna()
    x = data[var1_col]
    y = data[var2_col]
    n = len(data)
    
    if n < 3:
        return f"错误：样本量过少 (n={n})，无法计算相关系数。"
    
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        return f"错误：相关性分析仅适用于数值变量。\n检测到变量类型：\n- {var1_col}: {x.dtype}\n- {var2_col}: {y.dtype}"

    # 2. 正态性检验 (决定使用Pearson还是Spearman)
    # 如果样本量 > 50 使用 Kolmogorov-Smirnov，否则使用 Shapiro-Wilk
    def check_normality(series):
        if len(series) > 50:
            _, p = stats.kstest(series, 'norm')
        else:
            _, p = stats.shapiro(series)
        return p > 0.05

    norm1 = check_normality(x)
    norm2 = check_normality(y)
    
    use_pearson = norm1 and norm2
    
    # 3. 计算相关系数
    if use_pearson:
        r, p_val = stats.pearsonr(x, y)
        method_name = "Pearson相关系数 (Pearson Correlation)"
        desc = "数据符合正态分布，采用参数检验。"
    else:
        r, p_val = stats.spearmanr(x, y)
        method_name = "Spearman秩相关系数 (Spearman Rank Correlation)"
        desc = "数据不符合正态分布，采用非参数检验。"
        
    # 4. 强度判定
    abs_r = abs(r)
    if abs_r < 0.3: strength = "极弱相关或无相关"
    elif abs_r < 0.5: strength = "低度相关"
    elif abs_r < 0.8: strength = "中度相关"
    else: strength = "高度相关"
    
    if r > 0: direction = "正相关 (Positive)"
    elif r < 0: direction = "负相关 (Negative)"
    else: direction = "无"

    # 5. 生成报告
    report = f"=== {method_name} ===\n\n"
    report += f"变量 1: {var1_col}\n"
    report += f"变量 2: {var2_col}\n"
    report += f"样本量 (n): {n}\n\n"
    
    report += "1. 前提检验:\n"
    report += f"   - {var1_col} 正态性: {'是' if norm1 else '否'}\n"
    report += f"   - {var2_col} 正态性: {'是' if norm2 else '否'}\n"
    report += f"   - 决策: {desc}\n\n"
    
    report += "2. 分析结果:\n"
    report += f"   - 相关系数 (r) = {r:.3f}\n"
    report += f"   - P值 (Sig.) = {p_val:.4f} " + ("(***)" if p_val < 0.001 else "(**)" if p_val < 0.01 else "(*)" if p_val < 0.05 else "(ns)") + "\n\n"
    
    if p_val < 0.05:
        report += "3. 结论:\n"
        report += f"   两个变量之间存在显著的 **{strength}**。\n"
        report += f"   方向为: {direction}。\n"
        if use_pearson:
            report += f"   (解释: {var1_col} 解释了 {var2_col} 约 {r**2:.1%} 的变异 (R²))"
    else:
        report += "3. 结论:\n"
        report += "   两个变量之间不存在显著的线性相关关系 (P >= 0.05)。"
        
    return report
