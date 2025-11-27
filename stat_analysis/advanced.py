import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_pca_analysis(df, columns, n_components=None):
    """
    执行主成分分析 (PCA)
    :param df: 原始数据
    :param columns: 参与分析的数值变量列表
    :param n_components: 保留的主成分个数 (None则默认保留累计贡献>85%或全部)
    :return: 
        - report: 文本报告
        - pca_df: 降维后的数据 (前2个主成分用于绘图)
        - explained_variance: 解释方差比
        - components_df: 因子载荷矩阵
    """
    # 1. 数据准备与清洗
    data = df[columns].dropna()
    if len(data) < 2:
        return "错误：有效样本量不足", None, None, None
        
    X = data.values
    
    # 2. 标准化 (Z-score) - PCA的前提
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 运行 PCA
    # 先计算所有主成分以查看碎石图
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # 决定保留几个主成分 (简单的自动策略：特征值>1 或 累计>85%)
    # 这里为了简化，如果用户没指定，默认取前2个用于展示，或者全部
    if n_components is None:
        n_components = min(len(columns), 5) # 默认最多显示前5个
        
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. 生成报告
    report = []
    report.append(f"=== 主成分分析 (PCA) 结果 ===")
    report.append(f"样本量: {len(data)}")
    report.append(f"变量数: {len(columns)}")
    report.append("")
    
    report.append("【方差解释 (Eigenvalues)】")
    report.append(f"{'成分':<10} {'特征值':<10} {'方差贡献率(%)':<15} {'累计贡献率(%)':<15}")
    report.append("-" * 60)
    
    cum_var = 0
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        eigenvalue = pca.explained_variance_[i]
        var_pct = var_ratio * 100
        cum_var += var_pct
        report.append(f"PC{i+1:<8} {eigenvalue:<10.4f} {var_pct:<15.2f} {cum_var:<15.2f}")
    report.append("-" * 60)
    report.append("")
    
    report.append("【成分矩阵 (Component Matrix) - 载荷】")
    header = f"{'变量':<15} " + " ".join([f"PC{i+1:<8}" for i in range(n_components)])
    report.append(header)
    report.append("-" * (15 + 10 * n_components))
    
    components = pca.components_.T # 转置，行是变量，列是成分
    for i, col_name in enumerate(columns):
        row_str = f"{col_name:<15} "
        for j in range(n_components):
            # 载荷值
            load_val = components[i, j]
            # 简单的载荷高亮标记 (只是为了阅读，不影响逻辑)
            mark = "*" if abs(load_val) > 0.5 else " "
            row_str += f"{load_val:>8.4f}{mark} "
        report.append(row_str)
    report.append("-" * 60)
    report.append("注: * 表示载荷绝对值 > 0.5 (主要贡献变量)")
    
    # 准备绘图数据
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    # 如果有2个以上成分，我们只取前两个画二维散点图
    
    return "\n".join(report), pca_df, pca.explained_variance_ratio_, pd.DataFrame(components, index=columns, columns=[f'PC{i+1}' for i in range(n_components)])


def run_kmeans_clustering(df, columns, k=3):
    """
    执行 K-Means 聚类
    :param df: 数据
    :param columns: 参与聚类的变量
    :param k: 聚类簇数
    :return: 报告文本, 带有'Cluster'列的DataFrame(部分列), 聚类中心
    """
    # 1. 数据
    data_raw = df[columns].dropna()
    if len(data_raw) < k:
        return f"错误：样本量 ({len(data_raw)}) 小于聚类数 ({k})", None, None
        
    X = data_raw.values
    
    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # 4. 整理结果
    result_df = data_raw.copy()
    result_df['Cluster'] = labels + 1 # 让类别从1开始
    
    # 5. 生成报告
    report = []
    report.append(f"=== K-Means 聚类分析 (K={k}) ===")
    report.append(f"样本量: {len(data_raw)}")
    report.append(f"迭代次数: {kmeans.n_iter_}")
    report.append(f"惯性 (Inertia/组内平方和): {kmeans.inertia_:.2f}")
    report.append("")
    
    report.append("【聚类中心 (Cluster Centers) - 原始尺度】")
    # 需要将中心点还原回原始尺度以便理解
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 表头
    header = f"{'类别':<8} {'样本数':<8} " + " ".join([f"{c[:8]:<10}" for c in columns])
    report.append(header)
    report.append("-" * len(header))
    
    # 统计每个类的数量
    counts = pd.Series(labels).value_counts().sort_index()
    
    for i in range(k):
        n_samples = counts.get(i, 0)
        row_str = f"Class {i+1:<2} {n_samples:<8} "
        for val in centers_original[i]:
            row_str += f"{val:<10.2f} "
        report.append(row_str)
        
    report.append("-" * len(header))
    
    return "\n".join(report), result_df, centers_original
