import os
import pandas as pd
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, QTabWidget, 
                             QMessageBox, QTextEdit, QComboBox, QDialog, 
                             QFormLayout, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette

from .data_view import DataView
from .plot_view import PlotView
from .cleaning_dialog import CleaningDialog
from .activation_dialog import ActivationDialog
from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner
from core.license_manager import LicenseManager
from stat_analysis.descriptive import calculate_descriptive_stats
from stat_analysis.ttest import independent_ttest
from stat_analysis.anova import one_way_anova
from stat_analysis.correlation import correlation_analysis
from stat_analysis.regression import simple_linear_regression
from stat_analysis.advanced import run_pca_analysis, run_kmeans_clustering
from visualization.basic_plots import plot_distribution, plot_ttest_result, plot_anova_result, plot_correlation_result, plot_regression_result
from visualization.advanced_plots import plot_pca_scatter, plot_kmeans_scatter
from utils.helpers import resource_path

# --- 样式表配置 ---
STYLESHEET = """
QMainWindow {
    background-color: #F0F2F5;
}

/* 侧边栏样式 */
QWidget#SideBar {
    background-color: #2C3E50;
    color: white;
    border-right: 1px solid #1A252F;
}

QLabel#AppTitle {
    font-family: 'Segoe UI', 'Microsoft YaHei';
    font-size: 22px;
    font-weight: bold;
    color: #ECF0F1;
    padding: 20px 10px;
    margin-bottom: 10px;
}

QLabel#SectionTitle {
    color: #95A5A6;
    font-weight: bold;
    padding: 5px 10px;
    margin-top: 15px;
    font-size: 12px;
}

/* 侧边栏按钮 */
QPushButton.SideBtn {
    text-align: left;
    padding: 12px 20px;
    border: none;
    border-radius: 6px;
    background-color: transparent;
    color: #ECF0F1;
    font-size: 14px;
    margin: 2px 10px;
}

QPushButton.SideBtn:hover {
    background-color: #34495E;
}

QPushButton.SideBtn:checked {
    background-color: #4472C4;
    color: white;
    font-weight: bold;
    border-radius: 6px;
    background-color: #4472C4;
}

QPushButton.SideBtn:disabled {
    color: #7F8C8D;
}

QPushButton#BtnLoad {
    background-color: #4472C4;
    color: white;
    font-weight: bold;
    border-radius: 6px;
    padding: 12px;
    margin: 10px 10px 20px 10px;
    text-align: center;
}

QPushButton#BtnLoad:hover {
    background-color: #355C9E;
}

/* 内容区样式 */
QTabWidget::pane {
    border: 1px solid #E0E0E0;
    background: white;
    border-radius: 4px;
    top: -1px; 
}

QTabBar::tab {
    background: #E8E8E8;
    border: 1px solid #C4C4C3;
    border-bottom-color: #C4C4C3;
    min-width: 100px;
    padding: 8px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: white;
    border-color: #E0E0E0;
    border-bottom-color: white; 
    font-weight: bold;
    color: #4472C4;
}

QTextEdit {
    border: none;
    background-color: white;
    padding: 10px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.5;
}
"""

class SelectionDialog(QDialog):
    """通用参数选择对话框 (用于T检验/ANOVA/相关性)"""
    def __init__(self, columns, title="参数设置", parent=None, labels=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        
        # 表单区域
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        self.group_combo = QComboBox()
        self.group_combo.addItems(columns)
        self.value_combo = QComboBox()
        self.value_combo.addItems(columns)
        
        # 默认标签
        label1 = "分组变量 (Group):"
        label2 = "检验变量 (Value):"
        
        if labels:
            label1, label2 = labels
        
        # 智能预选
        # (对于相关性，这里可能不太准，但用户可以自己改)
        for col in columns:
            if 'group' in col.lower() or '组' in col:
                self.group_combo.setCurrentText(col)
            if 'score' in col.lower() or '值' in col or '量' in col:
                self.value_combo.setCurrentText(col)
        
        form_layout.addRow(label1, self.group_combo)
        form_layout.addRow(label2, self.value_combo)
        
        layout.addLayout(form_layout)
        
        layout.addSpacing(20)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("开始分析")
        btn_ok.setStyleSheet("""
            background-color: #4472C4; color: white; padding: 8px 20px; 
            border-radius: 4px; font-weight: bold;
        """)
        btn_ok.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet("padding: 8px 20px; border: 1px solid #ccc; border-radius: 4px;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        
        layout.addLayout(btn_layout)

    def get_selection(self):
        return self.group_combo.currentText(), self.value_combo.currentText()


class MultiSelectionDialog(QDialog):
    """多选对话框 (用于PCA)"""
    def __init__(self, columns, title="参数设置", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("请选择参与分析的变量 (建议选择数值型):"))
        
        # 列表框
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        
        for col in columns:
            item = QListWidgetItem(col)
            self.list_widget.addItem(item)
            # 默认不全选，让用户自己点
            
        layout.addWidget(self.list_widget)
        
        # 提示
        layout.addWidget(QLabel("按住 Ctrl 或 Shift 可多选"))
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("开始分析")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        
        layout.addLayout(btn_layout)
        
    def get_selection(self):
        return [item.text() for item in self.list_widget.selectedItems()]

class ClusterDialog(QDialog):
    """聚类参数对话框"""
    def __init__(self, columns, title="聚类设置", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        
        # 变量选择
        layout.addWidget(QLabel("选择聚类变量:"))
        from PyQt5.QtWidgets import QListWidget, QAbstractItemView, QSpinBox
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for col in columns:
            self.list_widget.addItem(col)
        layout.addWidget(self.list_widget)
        
        # K值选择
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("聚类数量 (K):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 10)
        self.spin_k.setValue(3)
        k_layout.addWidget(self.spin_k)
        k_layout.addStretch()
        layout.addLayout(k_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("开始聚类")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        
        layout.addLayout(btn_layout)

    def get_selection(self):
        cols = [item.text() for item in self.list_widget.selectedItems()]
        k = self.spin_k.value()
        return cols, k

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StatEase - 简易统计分析助手")
        self.resize(1280, 850)
        
        # 设置图标
        icon_path = resource_path("assets/icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 应用样式
        self.setStyleSheet(STYLESHEET)
        
        # 核心数据
        self.df = None
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        
        # 授权管理
        self.license_manager = LicenseManager()
        self.is_activated, self.license_msg, self.days_left = self.license_manager.check_license()
        
        # 根据授权状态调整标题
        if self.is_activated:
            self.setWindowTitle(f"StatEase - 专业版 ({self.license_msg})")
        else:
            self.setWindowTitle(f"StatEase - 免费试用版 (未激活)")
            
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === 1. 左侧侧边栏 ===
        sidebar = QWidget()
        sidebar.setObjectName("SideBar")
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(5)
        
        # Logo区域
        app_title = QLabel("StatEase")
        app_title.setObjectName("AppTitle")
        app_title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(app_title)
        
        # 导入按钮
        self.btn_load = QPushButton("📂  导入数据 (Excel)")
        self.btn_load.setObjectName("BtnLoad")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.load_file)
        sidebar_layout.addWidget(self.btn_load)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #34495E; margin: 10px;")
        sidebar_layout.addWidget(line)
        
        # 工具箱菜单
        sidebar_layout.addWidget(QLabel("数据工具", objectName="SectionTitle"))
        
        self.btn_clean = QPushButton("🧹  数据清洗与检测 (Pro)")
        self.setup_sidebar_btn(self.btn_clean, self.show_cleaning_dialog)
        sidebar_layout.addWidget(self.btn_clean)
        
        # 统计分析菜单
        sidebar_layout.addWidget(QLabel("统计分析", objectName="SectionTitle"))
        
        self.btn_desc = QPushButton("📊  描述性统计")
        self.setup_sidebar_btn(self.btn_desc, self.show_descriptive_stats)
        sidebar_layout.addWidget(self.btn_desc)
        
        self.btn_ttest = QPushButton("⚖️  两组比较 (T检验)")
        self.setup_sidebar_btn(self.btn_ttest, self.show_ttest_dialog)
        sidebar_layout.addWidget(self.btn_ttest)
        
        self.btn_anova = QPushButton("📊  多组比较 (ANOVA) (Pro)")
        self.setup_sidebar_btn(self.btn_anova, self.show_anova_dialog)
        sidebar_layout.addWidget(self.btn_anova)
        
        self.btn_corr = QPushButton("📈  相关性分析")
        self.setup_sidebar_btn(self.btn_corr, self.show_correlation_dialog)
        sidebar_layout.addWidget(self.btn_corr)
        
        self.btn_reg = QPushButton("📉  线性回归分析 (Pro)")
        self.setup_sidebar_btn(self.btn_reg, self.show_regression_dialog)
        sidebar_layout.addWidget(self.btn_reg)
        
        # 高级分析菜单
        sidebar_layout.addWidget(QLabel("高级分析", objectName="SectionTitle"))
        
        self.btn_pca = QPushButton("🧬  主成分分析 (Pro)")
        self.setup_sidebar_btn(self.btn_pca, self.show_pca_dialog)
        sidebar_layout.addWidget(self.btn_pca)
        
        self.btn_kmeans = QPushButton("🧩  K-Means 聚类 (Pro)")
        self.setup_sidebar_btn(self.btn_kmeans, self.show_kmeans_dialog)
        sidebar_layout.addWidget(self.btn_kmeans)
        
        # 实用工具菜单
        sidebar_layout.addWidget(QLabel("实用工具", objectName="SectionTitle"))
        
        self.btn_export = QPushButton("💾  导出分析报告 (Pro)")
        self.setup_sidebar_btn(self.btn_export, self.export_report)
        sidebar_layout.addWidget(self.btn_export)
        
        sidebar_layout.addStretch()
        
        # 激活按钮 (仅在未激活时显示)
        if not self.is_activated:
            btn_activate = QPushButton("🔑  激活专业版")
            btn_activate.setStyleSheet("""
                QPushButton {
                    background-color: #E67E22; color: white; font-weight: bold;
                    border-radius: 6px; padding: 10px; margin: 10px;
                }
                QPushButton:hover { background-color: #D35400; }
            """)
            btn_activate.setCursor(Qt.PointingHandCursor)
            btn_activate.clicked.connect(self.show_activation_dialog)
            sidebar_layout.addWidget(btn_activate)
        
        # 底部版本号
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet("color: #7F8C8D; padding: 10px; font-size: 11px;")
        version_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(version_label)
        
        # === 2. 右侧内容区 ===
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # 欢迎/状态栏
        self.status_bar = QLabel("欢迎使用 StatEase，请先导入数据开始分析。")
        self.status_bar.setStyleSheet("color: #555; font-size: 14px; margin-bottom: 10px;")
        content_layout.addWidget(self.status_bar)
        
        # 标签页
        self.tabs = QTabWidget()
        
        # 数据视图
        self.data_view = DataView()
        self.tabs.addTab(self.data_view, "📋 数据视图")
        
        # 结果视图
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("统计分析结果将显示在这里...")
        self.tabs.addTab(self.result_view, "📈 分析结果")
        
        # 图表视图 (新增)
        self.plot_view = PlotView()
        self.tabs.addTab(self.plot_view, "📊 图表展示")
        
        content_layout.addWidget(self.tabs)
        
        # 添加到主布局
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area)

    def setup_sidebar_btn(self, btn, func):
        """配置侧边栏按钮通用属性"""
        btn.setProperty("class", "SideBtn")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setEnabled(False)
        btn.clicked.connect(func)

    def show_activation_dialog(self):
        dialog = ActivationDialog(self.license_manager, self)
        if dialog.exec_() == QDialog.Accepted:
            # 激活成功，重启程序提示
            QMessageBox.information(self, "提示", "激活成功！请重启软件以解锁全部功能。")
            # 这里也可以选择动态刷新界面，简单起见建议重启

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)")
        if file_name:
            try:
                self.df = self.data_loader.load_file(file_name)
                self.data_view.load_data(self.df)
                
                # 激活按钮
                self.btn_clean.setEnabled(True)
                self.btn_desc.setEnabled(True)
                self.btn_ttest.setEnabled(True)
                self.btn_anova.setEnabled(True)
                self.btn_corr.setEnabled(True)
                self.btn_reg.setEnabled(True)
                self.btn_pca.setEnabled(True)
                self.btn_kmeans.setEnabled(True)
                self.btn_export.setEnabled(True)
                
                # 免费版：Pro功能变灰或者样式区分
                if not self.is_activated:
                    # 这里我们不禁用按钮，而是允许点击，点击后提示升级
                    # 为了用户体验，可以先不禁用，点击时拦截
                    pass
                
                self.btn_clean.setStyleSheet("")
                self.btn_desc.setStyleSheet("") 
                self.btn_ttest.setStyleSheet("")
                self.btn_anova.setStyleSheet("")
                self.btn_corr.setStyleSheet("")
                self.btn_reg.setStyleSheet("")
                self.btn_pca.setStyleSheet("")
                self.btn_kmeans.setStyleSheet("")
                self.btn_export.setStyleSheet("")
                
                # 更新状态
                filename_short = os.path.basename(file_name)
                self.status_bar.setText(f"当前文件: {filename_short} (共 {self.df.shape[0]} 行, {self.df.shape[1]} 列)")
                self.tabs.setCurrentIndex(0)
                
            except Exception as e:
                QMessageBox.critical(self, "加载失败", str(e))

    def check_pro_feature(self):
        """检查是否允许使用Pro功能"""
        if self.is_activated:
            return True
        
        QMessageBox.warning(self, "功能受限", 
            "这是专业版功能。\n\n"
            "免费版仅支持：\n"
            "✅ 数据导入与预览\n"
            "✅ 描述性统计\n"
            "✅ T检验\n"
            "✅ 相关性分析\n\n"
            "请激活专业版以解锁：\n"
            "🔒 数据清洗与检测\n"
            "🔒 ANOVA 方差分析\n"
            "🔒 线性回归分析\n"
            "🔒 主成分分析 (PCA)\n"
            "🔒 K-Means 聚类\n"
            "🔒 导出分析报告\n"
            "🔒 更多高级功能...")
        return False

    def show_pca_dialog(self):
        if self.df is None: return
        if not self.check_pro_feature(): return
        
        # 筛选数值列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "数据不足", "PCA 至少需要2个数值型变量")
            return
            
        dialog = MultiSelectionDialog(numeric_cols, "主成分分析 (PCA) 设置", self)
        if dialog.exec_() == QDialog.Accepted:
            cols = dialog.get_selection()
            if len(cols) < 2:
                QMessageBox.warning(self, "选择过少", "请至少选择2个变量进行降维")
                return
                
            try:
                # 运行分析
                report, pca_df, variance_ratio, components_df = run_pca_analysis(self.df, cols)
                if "错误" in report:
                    QMessageBox.warning(self, "分析错误", report)
                    return
                    
                self.result_view.setText(report)
                
                # 绘图 (前两个主成分)
                fig = plot_pca_scatter(pca_df, variance_ratio)
                self.plot_view.show_figure(fig)
                self.tabs.setCurrentIndex(2)
                
                self.status_bar.setText(f"PCA分析完成: {len(cols)} 个变量")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"PCA执行出错:\n{str(e)}")

    def show_kmeans_dialog(self):
        if self.df is None: return
        if not self.check_pro_feature(): return
        
        # 筛选数值列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            QMessageBox.warning(self, "数据不足", "聚类至少需要2个数值型变量")
            return
            
        dialog = ClusterDialog(numeric_cols, "K-Means 聚类设置", self)
        if dialog.exec_() == QDialog.Accepted:
            cols, k = dialog.get_selection()
            if len(cols) < 1:
                QMessageBox.warning(self, "未选择变量", "请至少选择1个变量进行聚类")
                return
                
            try:
                # 运行分析
                report, result_df, centers = run_kmeans_clustering(self.df, cols, k)
                if report.startswith("错误"):
                    QMessageBox.warning(self, "分析错误", report)
                    return
                
                self.result_view.setText(report)
                
                # 绘图 (如果有2个以上变量，取前两个画图)
                # 这里我们简单地取用户选的前两个，如果只有一个，那就没法画散点图了
                if len(cols) >= 2:
                    fig = plot_kmeans_scatter(result_df, cols[0], cols[1])
                    self.plot_view.show_figure(fig)
                    self.tabs.setCurrentIndex(2)
                else:
                    self.tabs.setCurrentIndex(1)
                    QMessageBox.information(self, "提示", "变量少于2个，未生成散点图。")
                
                self.status_bar.setText(f"K-Means聚类完成: K={k}")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"聚类执行出错:\n{str(e)}")

    def export_report(self):
        """导出分析结果为文件"""
        if not self.check_pro_feature(): return
        
        content = self.result_view.toPlainText()
        if not content or "统计分析结果将显示在这里" in content:
            QMessageBox.warning(self, "提示", "当前没有分析结果可导出。")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "导出分析报告", "Analysis_Report.txt", "Text Files (*.txt);;Markdown (*.md)")
        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(content)
                QMessageBox.information(self, "成功", f"报告已保存至：\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    def show_cleaning_dialog(self):
        if self.df is None: return
        if not self.check_pro_feature(): return
        
        # 1. 进行体检
        report = self.data_cleaner.check_quality(self.df)
        
        # 2. 高亮显示问题行 (红色: 重复, 橙色: 缺失)
        # 为了避免颜色冲突，如果某行既重复又缺失，优先显示重复(因为通常会先删重复)
        # 这里的颜色需要QColor
        red_color = QColor(255, 200, 200)
        orange_color = QColor(255, 230, 200)
        
        # 先清除旧高亮 (这里 DataView.highlight_rows 简单实现是覆盖，所以如果需要彻底清除，最好重载数据)
        # 暂时不重载，直接覆盖
        
        if report['missing_indices']:
            self.data_view.highlight_rows(report['missing_indices'], orange_color)
            
        if report['duplicate_indices']:
            self.data_view.highlight_rows(report['duplicate_indices'], red_color)
            
        # 切换到数据视图让用户看到高亮
        self.tabs.setCurrentIndex(0)
        
        # 3. 显示弹窗
        dialog = CleaningDialog(report, self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            
            # 4. 应用清洗
            try:
                new_df, logs = self.data_cleaner.apply_cleaning(self.df, config)
                self.df = new_df
                
                # 5. 刷新界面 (这会清除所有高亮，恢复正常)
                self.data_view.load_data(self.df)
                self.tabs.setCurrentIndex(0)
                
                # 显示日志
                log_text = "=== 数据清洗执行日志 ===\n\n" + ("\n".join(logs) if logs else "没有执行任何更改。")
                self.result_view.setText(log_text)
                self.tabs.setCurrentIndex(1)
                
                self.status_bar.setText(f"数据清洗完成，当前行数: {len(self.df)}")
                
            except Exception as e:
                QMessageBox.critical(self, "清洗失败", str(e))
        else:
            # 如果用户取消，最好清除高亮
            # 最简单的方法是重新加载数据
            self.data_view.load_data(self.df)
                
    def show_descriptive_stats(self):
        if self.df is None: return
        try:
            # 1. 生成文本报告
            stats_text = calculate_descriptive_stats(self.df)
            self.result_view.setText(stats_text)
            
            # 2. 生成图表
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            fig = plot_distribution(self.df, numeric_cols)
            self.plot_view.show_figure(fig)
            
            # 3. 切换到结果视图
            self.tabs.setCurrentIndex(1)
            self.status_bar.setText("描述性统计分析完成，请查看分析结果和图表展示。")
            
        except Exception as e:
            QMessageBox.warning(self, "分析错误", str(e))

    def show_ttest_dialog(self):
        if self.df is None: return
        
        dialog = SelectionDialog(self.df.columns, "独立样本 T检验设置", self)
        if dialog.exec_() == QDialog.Accepted:
            group_col, value_col = dialog.get_selection()
            if group_col == value_col:
                QMessageBox.warning(self, "输入错误", "分组变量和检验变量不能相同！")
                return
                
            try:
                # 文本报告
                report = independent_ttest(self.df, group_col, value_col)
                self.result_view.setText(report)
                
                # 图表
                group_names = self.df[group_col].dropna().unique()
                if len(group_names) == 2:
                    g1 = self.df[self.df[group_col] == group_names[0]][value_col].dropna()
                    g2 = self.df[self.df[group_col] == group_names[1]][value_col].dropna()
                    _, p_levene = stats.levene(g1, g2)
                    equal_var = p_levene > 0.05
                    _, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                    
                    fig = plot_ttest_result(self.df, group_col, value_col, p_val)
                    self.plot_view.show_figure(fig)
                    self.tabs.setCurrentIndex(2) 
                    self.status_bar.setText(f"T检验分析完成: {group_col} 对 {value_col} 的影响")
                else:
                    self.tabs.setCurrentIndex(1)
                    self.status_bar.setText(f"T检验分析完成 (注意: 组数不等于2，未生成T检验图表)")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"执行T检验时发生错误:\n{str(e)}")

    def show_anova_dialog(self):
        if self.df is None: return
        if not self.check_pro_feature(): return
        
        dialog = SelectionDialog(self.df.columns, "单因素方差分析 (One-way ANOVA) 设置", self)
        if dialog.exec_() == QDialog.Accepted:
            group_col, value_col = dialog.get_selection()
            if group_col == value_col:
                QMessageBox.warning(self, "输入错误", "分组变量和检验变量不能相同！")
                return
                
            try:
                # 文本报告
                report = one_way_anova(self.df, group_col, value_col)
                self.result_view.setText(report)
                
                # 图表 (使用f_oneway计算简单的P值用于绘图)
                group_data = [self.df[self.df[group_col] == g][value_col].dropna() for g in self.df[group_col].dropna().unique()]
                if len(group_data) > 1:
                    _, p_val = stats.f_oneway(*group_data)
                    fig = plot_anova_result(self.df, group_col, value_col, p_val)
                    self.plot_view.show_figure(fig)
                    self.tabs.setCurrentIndex(2)
                else:
                    self.tabs.setCurrentIndex(1)
                
                self.status_bar.setText(f"ANOVA分析完成: {group_col} 对 {value_col} 的影响")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"执行ANOVA时发生错误:\n{str(e)}")

    def show_correlation_dialog(self):
        if self.df is None: return
        
        dialog = SelectionDialog(self.df.columns, "相关性分析设置", self, labels=("变量 1 (X):", "变量 2 (Y):"))
        if dialog.exec_() == QDialog.Accepted:
            var1, var2 = dialog.get_selection()
            if var1 == var2:
                QMessageBox.warning(self, "输入错误", "请选择两个不同的变量！")
                return
                
            try:
                # 文本报告
                report = correlation_analysis(self.df, var1, var2)
                self.result_view.setText(report)
                
                # 图表 (需要简单的逻辑判断以决定绘图，这里我们简单地计算一次Pearson P值用于绘图标记)
                # 但为了准确，应该复用后端逻辑返回的值。不过当前架构没有分离得那么好，所以我们这里只负责简单绘图
                # 如果是数值变量，就画
                d1 = self.df[var1]
                d2 = self.df[var2]
                
                if np.issubdtype(d1.dtype, np.number) and np.issubdtype(d2.dtype, np.number):
                    # 这里简单用Pearson，因为绘图里的拟合线也是线性的
                    r, p = stats.pearsonr(d1.dropna(), d2.dropna())
                    
                    fig = plot_correlation_result(self.df, var1, var2, r, p)
                    self.plot_view.show_figure(fig)
                    self.tabs.setCurrentIndex(2)
                else:
                    self.tabs.setCurrentIndex(1)
                
                self.status_bar.setText(f"相关性分析完成: {var1} vs {var2}")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"执行相关性分析时发生错误:\n{str(e)}")

    def show_regression_dialog(self):
        if self.df is None: return
        if not self.check_pro_feature(): return
        
        dialog = SelectionDialog(self.df.columns, "简单线性回归设置", self, labels=("自变量 (X):", "因变量 (Y):"))
        if dialog.exec_() == QDialog.Accepted:
            x_col, y_col = dialog.get_selection()
            if x_col == y_col:
                QMessageBox.warning(self, "输入错误", "自变量和因变量不能相同！")
                return
            
            # 检查变量类型
            if not np.issubdtype(self.df[x_col].dtype, np.number) or not np.issubdtype(self.df[y_col].dtype, np.number):
                QMessageBox.warning(self, "类型错误", "回归分析仅支持数值型变量！")
                return
                
            try:
                # 文本报告
                report = simple_linear_regression(self.df, x_col, y_col)
                if "错误" in report and "样本量" in report:
                     QMessageBox.warning(self, "数据错误", report)
                     return
                     
                self.result_view.setText(report)
                
                # 图表
                fig = plot_regression_result(self.df, x_col, y_col)
                self.plot_view.show_figure(fig)
                self.tabs.setCurrentIndex(2)
                
                self.status_bar.setText(f"回归分析完成: {y_col} ~ {x_col}")
                
            except Exception as e:
                QMessageBox.critical(self, "分析失败", f"执行回归分析时发生错误:\n{str(e)}")
