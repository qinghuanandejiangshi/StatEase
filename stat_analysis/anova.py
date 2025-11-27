import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def one_way_anova(df, group_col, value_col):
    """
    执行单因素方差分析 (One-way ANOVA)
    :param df: DataFrame
    :param group_col: 分组变量
    :param value_col: 数值变量
    :return: 格式化的文本报告
    """
    # 1. 数据准备
    groups = df[group_col].dropna().unique()
    if len(groups) < 3:
        return f"提示：分组变量 '{group_col}' 仅包含 {len(groups)} 个组别。建议使用 T检验 进行两组比较。"
    
    # 提取各组数据
    group_data = []
    group_names = []
    for g in groups:
        data = df[df[group_col] == g][value_col].dropna()
        if len(data) < 2:
            return f"错误：组别 '{g}' 的样本量过少 (<2)，无法进行分析。"
        group_data.append(data)
        group_names.append(g)
        
    # 2. 描述性统计
    desc_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std']).reset_index()
    
    # 3. 方差齐性检验 (Levene Test)
    stat_lev, p_lev = stats.levene(*group_data)
    
    # 4. ANOVA 主效应检验
    f_stat, p_val = stats.f_oneway(*group_data)
    
    # 5. 事后检验 (Tukey HSD)
    # 只有当ANOVA显著(p<0.05)时才推荐看事后检验，但为了完整性这里总是计算
    tukey = pairwise_tukeyhsd(endog=df[value_col].dropna(), 
                              groups=df[group_col].dropna(), 
                              alpha=0.05)
    
    # 6. 生成报告
    report = "=== 单因素方差分析结果 (One-way ANOVA) ===\n\n"
    
    report += "1. 描述性统计:\n"
    report += desc_stats.to_string(index=False, float_format="%.2f") + "\n\n"
    
    report += "2. 方差齐性检验 (Levene's Test):\n"
    report += f"   - F = {stat_lev:.3f}, p = {p_lev:.3f}\n"
    if p_lev > 0.05:
        report += "   - 结论: 方差齐性 (p > 0.05)，ANOVA结果可靠。\n\n"
    else:
        report += "   - 结论: 方差不齐 (p <= 0.05)，建议谨慎参考ANOVA结果或使用非参数检验(Kruskal-Wallis)。\n\n"
        
    report += "3. ANOVA 主效应检验:\n"
    report += f"   - F = {f_stat:.3f}\n"
    report += f"   - p = {p_val:.4f} " + ("(***)" if p_val < 0.001 else "(**)" if p_val < 0.01 else "(*)" if p_val < 0.05 else "(ns)") + "\n"
    
    if p_val < 0.05:
        report += "   - 结论: 各组之间存在显著差异。\n\n"
        
        report += "4. Tukey HSD 事后多重比较:\n"
        # 格式化Tukey结果
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        # 筛选出显著的行
        sig_pairs = tukey_df[tukey_df['reject'] == True]
        
        if not sig_pairs.empty:
            report += "   (仅显示存在显著差异的组对)\n"
            for _, row in sig_pairs.iterrows():
                report += f"   - {row['group1']} vs {row['group2']}: diff={row['meandiff']:.2f}, p={row['p-adj']:.4f}\n"
        else:
            report += "   (未发现两两之间存在显著差异)\n"
            
        report += "\n" + str(tukey.summary())
    else:
        report += "   - 结论: 各组之间未发现显著差异，无需进行事后检验。"
        
    return report
