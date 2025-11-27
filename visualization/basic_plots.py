import matplotlib
matplotlib.use('Qt5Agg')  # 指定后端为 Qt5
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

# --- 全局绘图风格设置 (符合SCI论文要求) ---
def set_sci_style():
    # 设置字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans'] # 优先使用Arial，中文回退到黑体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    
    # 颜色和线条
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.3
    
    # 默认DPI
    plt.rcParams['figure.dpi'] = 100 
    
    # 配色方案 (专业学术蓝/橙/灰)
    sns.set_palette(['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'])

set_sci_style()

def create_figure(figsize=(8, 6)):
    """创建一个新的Figure对象用于嵌入Qt界面"""
    fig = Figure(figsize=figsize, dpi=100)
    # 必须给Figure设置这个，否则seaborn绘图可能会报错或找不到ax
    return fig

def plot_distribution(df, numeric_cols):
    """
    绘制描述性统计图表：箱线图 + 直方图
    :param df: 数据框
    :param numeric_cols: 数值列名列表
    :return: figure对象
    """
    n_cols = len(numeric_cols)
    if n_cols == 0:
        return None
    
    # 限制最多显示前9个变量，避免图表过于拥挤
    display_cols = numeric_cols[:9]
    n = len(display_cols)
    
    # 动态计算子图布局
    if n == 1:
        rows, cols = 1, 2
        fig = Figure(figsize=(10, 5), dpi=100)
        axes = fig.subplots(1, 2)
        # 左边箱线图
        sns.boxplot(y=df[display_cols[0]], ax=axes[0], color='#4472C4', width=0.4)
        axes[0].set_title(f"{display_cols[0]} - Boxplot")
        # 右边直方图
        sns.histplot(x=df[display_cols[0]], ax=axes[1], kde=True, color='#4472C4')
        axes[1].set_title(f"{display_cols[0]} - Histogram")
        
    else:
        # 如果有多个变量，只画箱线图对比
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.subplots(1, 1)
        
        # 数据重组为长格式以便绘图
        plot_data = df[display_cols].melt(var_name='Variable', value_name='Value')
        
        sns.boxplot(x='Variable', y='Value', data=plot_data, ax=ax, width=0.5, palette='viridis')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Distribution of Numeric Variables")
        
    fig.tight_layout()
    return fig

def plot_ttest_result(df, group_col, value_col, p_value):
    """
    绘制T检验结果：带误差线的柱状图
    :param df: 数据框
    :param group_col: 分组列
    :param value_col: 数值列
    :param p_value: 显著性P值
    :return: figure对象
    """
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.subplots(1, 1)
    
    # 计算均值和标准误 (SE)
    stats_data = df.groupby(group_col)[value_col].agg(['mean', 'sem', 'count']).reset_index()
    
    # 绘图
    bars = ax.bar(stats_data[group_col], stats_data['mean'], 
                  yerr=stats_data['sem'], 
                  capsize=10, 
                  color=['#4472C4', '#ED7D31'], 
                  alpha=0.8,
                  width=0.5,
                  edgecolor='black')
    
    # 设置标签
    ax.set_ylabel(f'Mean of {value_col} (+/- SE)')
    ax.set_xlabel(group_col)
    ax.set_title(f'Comparison of {value_col} by {group_col}')
    
    # 自动调整Y轴范围以容纳显著性标记
    y_max = stats_data['mean'].max() + stats_data['sem'].max()
    ax.set_ylim(0, y_max * 1.2)
    
    # 添加显著性标记
    # 简单的标记逻辑：在两组上方画线
    if len(stats_data) == 2:
        x1, x2 = 0, 1
        y_line = y_max * 1.1
        h = y_max * 0.02
        
        # 画横线
        ax.plot([x1, x1, x2, x2], [y_line-h, y_line, y_line, y_line-h], lw=1.5, c='k')
        
        # 标记符号
        significance = "ns"
        if p_value < 0.001: significance = "***"
        elif p_value < 0.01: significance = "**"
        elif p_value < 0.05: significance = "*"
            
        ax.text((x1+x2)*.5, y_line, significance, ha='center', va='bottom', color='k', fontsize=14, fontweight='bold')
        
    fig.tight_layout()
    return fig

def plot_anova_result(df, group_col, value_col, p_value):
    """
    绘制ANOVA结果：带误差线的柱状图
    :param df: 数据框
    :param group_col: 分组列
    :param value_col: 数值列
    :param p_value: ANOVA的显著性P值
    :return: figure对象
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.subplots(1, 1)
    
    # 计算均值和标准误 (SE)
    stats_data = df.groupby(group_col)[value_col].agg(['mean', 'sem', 'count']).reset_index()
    
    # 简单的配色循环
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']
    if len(stats_data) > len(colors):
        colors = colors * (len(stats_data) // len(colors) + 1)
    
    # 绘图
    bars = ax.bar(stats_data[group_col], stats_data['mean'], 
                  yerr=stats_data['sem'], 
                  capsize=10, 
                  color=colors[:len(stats_data)], 
                  alpha=0.8,
                  width=0.6,
                  edgecolor='black')
    
    # 设置标签
    ax.set_ylabel(f'Mean of {value_col} (+/- SE)')
    ax.set_xlabel(group_col)
    ax.set_title(f'Comparison of {value_col} by {group_col} (One-way ANOVA)')
    
    # 自动调整Y轴范围
    if not stats_data.empty:
        y_max = stats_data['mean'].max() + (stats_data['sem'].max() if not np.isnan(stats_data['sem'].max()) else 0)
        ax.set_ylim(0, y_max * 1.2)
    
    # 标注整体显著性
    title_suffix = "ns"
    if p_value < 0.001: title_suffix = "***"
    elif p_value < 0.01: title_suffix = "**"
    elif p_value < 0.05: title_suffix = "*"
    
    ax.text(0.98, 0.95, f"ANOVA p={p_value:.3f} ({title_suffix})", 
            transform=ax.transAxes, ha='right', va='top', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
    fig.tight_layout()
    return fig

def plot_correlation_result(df, var1, var2, r_val, p_val):
    """
    绘制相关性分析结果：散点图 + 拟合线
    :param df: 数据框
    :param var1: 变量1
    :param var2: 变量2
    :param r_val: 相关系数
    :param p_val: P值
    :return: figure对象
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.subplots(1, 1)
    
    data = df[[var1, var2]].dropna()
    
    # 绘制散点
    sns.scatterplot(x=var1, y=var2, data=data, ax=ax, 
                    color='#4472C4', alpha=0.7, s=100, edgecolor='w')
    
    # 绘制回归线
    sns.regplot(x=var1, y=var2, data=data, ax=ax, 
                scatter=False, color='#ED7D31', line_kws={'linestyle': '--'})
    
    # 设置标签
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f"Correlation: {var1} vs {var2}")
    
    # 标注R和P
    label = f"r = {r_val:.3f}\np = {p_val:.4f}"
    ax.text(0.05, 0.95, label, transform=ax.transAxes, 
            fontsize=12, va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ddd'))
    
    fig.tight_layout()
    return fig

def plot_regression_result(df, x_col, y_col):
    """
    绘制回归分析结果：散点图 + 强化的回归线展示
    :param df: 数据
    :param x_col: 自变量
    :param y_col: 因变量
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.subplots(1, 1)
    
    data = df[[x_col, y_col]].dropna()
    
    # 使用 seaborn 的 regplot 绘制，带有置信区间
    sns.regplot(x=x_col, y=y_col, data=data, ax=ax,
                scatter_kws={'color': '#4472C4', 'alpha': 0.6, 's': 80},
                line_kws={'color': '#ED7D31', 'linewidth': 2.5})
                
    ax.set_title(f"Simple Linear Regression: {y_col} ~ {x_col}")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    return fig
