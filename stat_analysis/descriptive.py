import pandas as pd
import numpy as np

def calculate_descriptive_stats(df):
    """
    计算描述性统计量
    :param df: pandas DataFrame
    :return: 格式化的文本报告
    """
    # 分离数值列和非数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    
    report = "=== 描述性统计报告 ===\n\n"
    
    # 1. 数值变量统计
    if len(numeric_cols) > 0:
        report += "--- 数值变量统计 ---\n"
        stats = df[numeric_cols].describe().T
        # 添加更多统计量 if needed, describe已经包含count, mean, std, min, 25%, 50%, 75%, max
        
        # 格式化输出
        # 使用pandas to_markdown (需要安装tabulate) 或者手动格式化
        # 这里手动简单的格式化，避免额外依赖
        
        header = f"{'变量名':<15} {'N':<6} {'均值':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10}\n"
        report += header
        report += "-" * len(header) + "\n"
        
        for idx, row in stats.iterrows():
            # 截断过长的变量名
            name = str(idx)
            if len(name) > 12: name = name[:12] + ".."
            
            line = f"{name:<15} {int(row['count']):<6} {row['mean']:.2f}      {row['std']:.2f}      {row['min']:.2f}      {row['max']:.2f}\n"
            report += line
        report += "\n"
        
    # 2. 分类变量统计
    if len(object_cols) > 0:
        report += "--- 分类变量统计 (频数/百分比) ---\n"
        for col in object_cols:
            report += f"\n变量: {col}\n"
            counts = df[col].value_counts()
            total = len(df)
            
            for category, count in counts.items():
                pct = (count / total) * 100
                report += f"  - {category}: {count} ({pct:.1f}%)\n"
                
    if len(numeric_cols) == 0 and len(object_cols) == 0:
        report += "未检测到有效数据列。\n"
        
    return report
