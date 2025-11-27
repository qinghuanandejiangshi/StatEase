import pandas as pd
import numpy as np

class DataCleaner:
    """数据清洗与质量检测核心模块"""
    
    def get_cols_to_check(self, df):
        """获取应该参与查重的列（排除ID列）"""
        # 简单的启发式规则：排除包含 'id', '编号', 'number' 且不仅包含重复值的列
        # 这里为了稳健，先不自动排除，而是提供给用户选择，或者默认全部。
        # 我们在 check_quality 里实现简单的自动排除逻辑供参考
        cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if 'id' in col_lower or '编号' in col_lower or '序号' in col_lower:
                # 如果这一列是唯一的（像是ID），往往不应该参与重复行检查
                if df[col].is_unique:
                    continue
            cols.append(col)
        return cols if cols else df.columns.tolist()

    def check_quality(self, df):
        """
        对数据进行健康检查
        :return: 包含检测结果的字典
        """
        # 智能推断查重列（排除可能的ID列）
        subset_cols = self.get_cols_to_check(df)
        
        # 1. 重复值检测
        # keep=False 标记所有重复的行，keep='first' 标记除第一个外的
        # 我们为了高亮，标记所有重复的组（包括第一行），这样用户能看到哪些是重复的
        dup_mask = df.duplicated(subset=subset_cols, keep=False)
        duplicate_indices = df[dup_mask].index.tolist()
        
        # 2. 缺失值检测
        missing_mask = df.isnull().any(axis=1)
        missing_indices = df[missing_mask].index.tolist()
        
        report = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'subset_cols': subset_cols, # 实际用于查重的列
            'duplicates': len(duplicate_indices), # 注意：这里统计的是所有涉嫌重复的行数
            'duplicate_indices': duplicate_indices,
            'missing_count': df.isnull().sum().sum(),
            'missing_indices': missing_indices,
            'missing_details': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            'outliers': {} # 仅检测数值列
        }
        
        # 3. 异常值检测
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_condition = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))
            n_outliers = outlier_condition.sum()
            if n_outliers > 0:
                report['outliers'][col] = n_outliers
                
        return report
    
    def apply_cleaning(self, df, config):
        """
        根据配置应用清洗策略
        :param df: 原始 DataFrame
        :param config: 清洗配置字典
        :return: 清洗后的 DataFrame, 操作日志
        """
        df_clean = df.copy()
        logs = []
        
        # 1. 处理重复值
        if config.get('remove_duplicates'):
            subset = config.get('duplicate_subset') # 如果没有指定，默认None(所有列)
            
            n_before = len(df_clean)
            # keep='first' 保留第一个，删除后面的
            df_clean.drop_duplicates(subset=subset, keep='first', inplace=True)
            n_dropped = n_before - len(df_clean)
            
            if n_dropped > 0:
                cols_str = "所有列" if subset is None else f"排除ID列 ({len(subset)}列)"
                logs.append(f"✅ 已删除 {n_dropped} 行重复数据 (依据: {cols_str})")
        
        # 2. 处理缺失值
        if config.get('handle_missing'):
            method = config.get('missing_method', 'mean') # mean, median, drop
            
            if method == 'drop':
                n_before = len(df_clean)
                df_clean.dropna(inplace=True)
                n_dropped = n_before - len(df_clean)
                if n_dropped > 0:
                    logs.append(f"✅ 已删除 {n_dropped} 行包含缺失值的记录")
            else:
                # 填充策略
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                filled_count = 0
                
                for col in df_clean.columns:
                    if df_clean[col].isnull().sum() > 0:
                        if col in numeric_cols:
                            if method == 'mean':
                                fill_val = df_clean[col].mean()
                                method_str = "均值"
                            else: # median
                                fill_val = df_clean[col].median()
                                method_str = "中位数"
                            df_clean[col].fillna(fill_val, inplace=True)
                        else:
                            # 非数值列默认用众数填充
                            if not df_clean[col].mode().empty:
                                fill_val = df_clean[col].mode()[0]
                            else:
                                fill_val = "Unknown"
                            method_str = "众数"
                            df_clean[col].fillna(fill_val, inplace=True)
                        filled_count += 1
                        
                if filled_count > 0:
                    logs.append(f"✅ 已对 {filled_count} 个列进行了缺失值填充 ({method_str})")

        # 重置索引，防止后续处理索引不连续
        df_clean.reset_index(drop=True, inplace=True)
        return df_clean, logs
