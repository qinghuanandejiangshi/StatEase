from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class ActivationDialog(QDialog):
    def __init__(self, license_manager, parent=None):
        super().__init__(parent)
        self.license_manager = license_manager
        self.setWindowTitle("激活 StatEase 专业版")
        self.setFixedSize(450, 250)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title = QLabel("请输入您的产品激活码")
        title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 输入框
        self.input_key = QLineEdit()
        self.input_key.setPlaceholderText("在此粘贴激活码...")
        self.input_key.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                font-size: 14px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
            }
            QLineEdit:focus {
                border: 2px solid #4472C4;
            }
        """)
        layout.addWidget(self.input_key)
        
        # 提示文字
        self.lbl_status = QLabel("购买请联系开发者获取激活码")
        self.lbl_status.setStyleSheet("color: #7F8C8D;")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_activate = QPushButton("立即激活")
        self.btn_activate.setStyleSheet("""
            QPushButton {
                background-color: #4472C4;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #355C9E;
            }
        """)
        self.btn_activate.clicked.connect(self.do_activate)
        
        self.btn_cancel = QPushButton("以后再说")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7F8C8D;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #34495E;
            }
        """)
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_activate)
        layout.addLayout(btn_layout)
        
    def do_activate(self):
        key = self.input_key.text().strip()
        if not key:
            self.lbl_status.setText("请输入激活码")
            self.lbl_status.setStyleSheet("color: red;")
            return
            
        success, msg = self.license_manager.activate(key)
        
        if success:
            QMessageBox.information(self, "激活成功", msg)
            self.accept()
        else:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("color: red;")
