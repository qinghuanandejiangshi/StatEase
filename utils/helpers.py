import sys
import os

def resource_path(relative_path):
    """ 
    获取资源的绝对路径
    用于解决 PyInstaller 打包后，资源文件被解压到临时目录(sys._MEIPASS)找不到的问题
    """
    try:
        # PyInstaller 创建临时文件夹，将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        # 正常运行模式
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
