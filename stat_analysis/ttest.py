import pandas as pd
from scipy import stats
import numpy as np

def independent_ttest(df, group_col, value_col):
    """
    执行独立样本T检验
    :param df: DataFrame
    :param group_col: 分组变量列名
    :param value_col: 检验变量列名 (数值)
    :return: 格式化的文本报告
    """
    # 1. 数据准备
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return f"错误：分组变量 '{group_col}' 必须包含且仅包含 2 个组别，当前发现 {len(groups)} 个: {groups}"
    
    group1_name = groups[0]
    group2_name = groups[1]
    
    data1 = df[df[group_col] == group1_name][value_col].dropna()
    data2 = df[df[group_col] == group2_name][value_col].dropna()
    
    if len(data1) < 2 or len(data2) < 2:
        return "错误：每个组别至少需要2个样本才能进行T检验"
        
    # 2. 描述性统计
    n1, m1, s1 = len(data1), np.mean(data1), np.std(data1, ddof=1)
    n2, m2, s2 = len(data2), np.mean(data2), np.std(data2, ddof=1)
    
    # 3. 方差齐性检验 (Levene Test)
    stat_lev, p_lev = stats.levene(data1, data2)
    equal_var = p_lev > 0.05
    
    # 4. T检验
    t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    # 5. 效应量 (Cohen's d)
    # 简单的合并标准差计算
    n_total = n1 + n2
    dof = n_total - 2
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / dof)
    cohens_d = (m1 - m2) / pooled_std
    
    # 6. 生成报告
    report = "=== 独立样本 T检验结果 (Independent Samples T-Test) ===\n\n"
    
    report += "1. 描述性统计:\n"
    report += f"   - {group1_name}: n={n1}, Mean={m1:.2f}, SD={s1:.2f}\n"
    report += f"   - {group2_name}: n={n2}, Mean={m2:.2f}, SD={s2:.2f}\n\n"
    
    report += "2. 方差齐性检验 (Levene's Test):\n"
    report += f"   - F={stat_lev:.3f}, p={p_lev:.3f}\n"
    if equal_var:
        report += "   - 结果: 方差齐性 (p > 0.05)，采用标准T检验\n\n"
    else:
        report += "   - 结果: 方差不齐 (p <= 0.05)，采用Welch's T检验\n\n"
        
    report += "3. T检验结果:\n"
    report += f"   - t = {t_stat:.3f}\n"
    report += f"   - p = {p_val:.4f} " + ("(***)" if p_val < 0.001 else "(**)" if p_val < 0.01 else "(*)" if p_val < 0.05 else "(ns)") + "\n"
    report += f"   - Cohen's d = {abs(cohens_d):.3f} " + ("(大)" if abs(cohens_d)>0.8 else "(中)" if abs(cohens_d)>0.5 else "(小)") + "\n\n"
    
    report += "4. 结论:\n"
    if p_val < 0.05:
        report += f"   {group1_name} 与 {group2_name} 在 '{value_col}' 上存在显著差异。"
    else:
        report += f"   {group1_name} 与 {group2_name} 在 '{value_col}' 上未发现显著差异。"
        
    return report
