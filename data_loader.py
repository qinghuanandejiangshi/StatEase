import pandas as pd
import os

class DataLoader:
    def load_file(self, file_path):
        """
        加载Excel或CSV文件
        :param file_path: 文件绝对路径
        :return: pandas DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in ['.xlsx', '.xls']:
                # 默认读取第一个sheet
                df = pd.read_excel(file_path)
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
                
            if df.empty:
                raise ValueError("文件为空")
                
            return df
            
        except Exception as e:
            raise Exception(f"读取文件出错: {str(e)}")
