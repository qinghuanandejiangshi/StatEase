from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QCheckBox, QRadioButton, 
                             QButtonGroup, QFormLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class CleaningDialog(QDialog):
    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据质量检测与清洗")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.report = report
        self.config = {}
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. 头部检测报告
        header_box = QGroupBox("📊 数据体检报告")
        header_layout = QVBoxLayout()
        
        # 汇总信息
        summary_text = f"总行数: {self.report['n_rows']} | 总列数: {self.report['n_cols']}"
        header_layout.addWidget(QLabel(summary_text))
        
        # 问题列表
        problems = []
        if self.report['duplicates'] > 0:
            # 增加显示依据
            ignored_cols = set(self.report['subset_cols']) ^ set(self.config.get('all_cols', [])) # 这里简化处理，只提示排除ID
            # 更好的方式是直接显示 subset_cols 的数量
            subset_len = len(self.report['subset_cols'])
            total_len = self.report['n_cols']
            msg = f"⚠️ 发现 {self.report['duplicates']} 行涉嫌重复的数据"
            if subset_len < total_len:
                msg += f" (已忽略ID列)"
            problems.append(msg)
        
        if self.report['missing_count'] > 0:
            problems.append(f"⚠️ 发现 {self.report['missing_count']} 个缺失值 (涉及 {len(self.report['missing_details'])} 列)")
            
        if len(self.report['outliers']) > 0:
             problems.append(f"ℹ️ 发现 {sum(self.report['outliers'].values())} 个潜在异常值 (基于IQR规则)")
             
        if not problems:
            good_label = QLabel("✅ 数据质量良好，未发现明显问题。")
            good_label.setStyleSheet("color: green; font-weight: bold;")
            header_layout.addWidget(good_label)
        else:
            for p in problems:
                lbl = QLabel(p)
                lbl.setStyleSheet("color: #D35400; font-weight: bold;")
                header_layout.addWidget(lbl)
                
        header_box.setLayout(header_layout)
        layout.addWidget(header_box)
        
        # 2. 清洗选项
        if self.report['duplicates'] > 0 or self.report['missing_count'] > 0:
            options_box = QGroupBox("🛠️ 清洗策略")
            options_layout = QVBoxLayout()
            
            # 重复值选项
            if self.report['duplicates'] > 0:
                self.chk_dupes = QCheckBox(f"删除重复行 (共 {self.report['duplicates']} 行)")
                self.chk_dupes.setChecked(True)
                options_layout.addWidget(self.chk_dupes)
                
                # 提示依据
                subset_len = len(self.report['subset_cols'])
                total_len = self.report['n_cols']
                if subset_len < total_len:
                    lbl_hint = QLabel(f"   * 查重依据: {subset_len} 个列 (已自动排除ID/编号列)")
                    lbl_hint.setStyleSheet("color: #7F8C8D; font-size: 11px;")
                    options_layout.addWidget(lbl_hint)
            
            # 缺失值选项
            if self.report['missing_count'] > 0:
                self.chk_missing = QCheckBox("处理缺失值")
                self.chk_missing.setChecked(True)
                options_layout.addWidget(self.chk_missing)
                
                # 缺失值处理方法的子选项
                self.missing_method_group = QButtonGroup(self)
                self.radio_mean = QRadioButton("数值列用均值填充 / 分类列用众数填充")
                self.radio_median = QRadioButton("数值列用中位数填充 / 分类列用众数填充")
                self.radio_drop = QRadioButton("直接删除包含缺失值的行")
                
                self.radio_mean.setChecked(True)
                self.missing_method_group.addButton(self.radio_mean)
                self.missing_method_group.addButton(self.radio_median)
                self.missing_method_group.addButton(self.radio_drop)
                
                missing_options_layout = QVBoxLayout()
                missing_options_layout.setContentsMargins(20, 0, 0, 0)
                missing_options_layout.addWidget(self.radio_mean)
                missing_options_layout.addWidget(self.radio_median)
                missing_options_layout.addWidget(self.radio_drop)
                
                options_layout.addLayout(missing_options_layout)
                
                # 关联启用状态
                self.chk_missing.toggled.connect(self.radio_mean.setEnabled)
                self.chk_missing.toggled.connect(self.radio_median.setEnabled)
                self.chk_missing.toggled.connect(self.radio_drop.setEnabled)
            
            options_box.setLayout(options_layout)
            layout.addWidget(options_box)
        
        layout.addStretch()
        
        # 3. 底部按钮
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("应用清洗")
        self.btn_apply.setStyleSheet("background-color: #4472C4; color: white; font-weight: bold; padding: 8px 15px;")
        self.btn_apply.clicked.connect(self.on_apply)
        
        self.btn_cancel = QPushButton("取消 / 暂不处理")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)
        
    def on_apply(self):
        # 收集配置
        if hasattr(self, 'chk_dupes'):
            self.config['remove_duplicates'] = self.chk_dupes.isChecked()
            # 传递查重依据列
            if self.chk_dupes.isChecked():
                self.config['duplicate_subset'] = self.report['subset_cols']
        else:
            self.config['remove_duplicates'] = False
            
        if hasattr(self, 'chk_missing') and self.report['missing_count'] > 0:
            self.config['handle_missing'] = self.chk_missing.isChecked()
            if self.radio_drop.isChecked():
                self.config['missing_method'] = 'drop'
            elif self.radio_median.isChecked():
                self.config['missing_method'] = 'median'
            else:
                self.config['missing_method'] = 'mean'
        else:
            self.config['handle_missing'] = False
            
        self.accept()
        
    def get_config(self):
        return self.config
