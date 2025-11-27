import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

# 复用 basic_plots 的样式设置
def set_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_palette(['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47'])

set_style()

def plot_pca_scatter(pca_df, explained_variance_ratio):
    """
    绘制 PCA 双标图 (Score Plot) - 仅展示前两个主成分
    :param pca_df: 包含 PC1, PC2... 的数据框
    :param explained_variance_ratio: 解释方差比列表
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.subplots(1, 1)
    
    # 提取方差贡献率
    var1 = explained_variance_ratio[0] * 100
    var2 = explained_variance_ratio[1] * 100
    
    # 绘制散点
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax, 
                    s=100, alpha=0.7, color='#4472C4', edgecolor='w')
    
    # 添加中心轴线
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    
    # 标签
    ax.set_xlabel(f'PC1 ({var1:.1f}%)')
    ax.set_ylabel(f'PC2 ({var2:.1f}%)')
    ax.set_title('PCA Score Plot (PC1 vs PC2)')
    
    # 简单的网格
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_kmeans_scatter(result_df, col_x, col_y):
    """
    绘制 K-Means 聚类结果图
    :param result_df: 包含原始变量和 'Cluster' 列的数据框
    :param col_x: X轴显示的变量名
    :param col_y: Y轴显示的变量名
    """
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.subplots(1, 1)
    
    # 确保 Cluster 是分类变量，以便 Seaborn 正确着色
    plot_data = result_df.copy()
    plot_data['Cluster'] = plot_data['Cluster'].astype(str)
    
    # 绘制散点，颜色映射到 Cluster
    sns.scatterplot(x=col_x, y=col_y, hue='Cluster', data=plot_data, ax=ax, 
                    palette='viridis', s=100, alpha=0.8, edgecolor='w')
    
    ax.set_title(f'K-Means Clustering Result ({col_y} vs {col_x})')
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig
