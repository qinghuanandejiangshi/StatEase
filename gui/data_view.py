from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt

class DataView(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(True)
        
    def load_data(self, df):
        """加载pandas DataFrame到表格中"""
        self.setRowCount(df.shape[0])
        self.setColumnCount(df.shape[1])
        self.setHorizontalHeaderLabels(df.columns.astype(str))
        
        # 为了性能，只显示前1000行（如果数据量大的话，后续可以优化为分页或懒加载）
        # MVP阶段简单处理，如果行数太多截断显示或者全部显示（注意性能）
        # 这里暂时显示全部，如果卡顿后续优化
        
        # 考虑到性能，如果超过1000行，先只显示前1000行并在表头提示
        display_rows = min(1000, df.shape[0])
        self.setRowCount(display_rows)
        
        for i in range(display_rows):
            for j in range(df.shape[1]):
                value = df.iloc[i, j]
                # 格式化显示
                if isinstance(value, float):
                    item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))
                
                item.setFlags(item.flags() ^ Qt.ItemIsEditable) # 只读
                self.setItem(i, j, item)
        
        if df.shape[0] > 1000:
            print(f"提示: 仅显示前1000行数据预览 (共 {df.shape[0]} 行)")

    def highlight_rows(self, row_indices, color):
        """
        高亮显示指定的行
        :param row_indices: 行索引列表
        :param color: QColor 对象
        """
        # 清除之前的背景色 (重置为交替色)
        # 但这里为了简单，只覆盖颜色。如果需要重置，应该重新load_data
        
        for row in row_indices:
            if row >= self.rowCount(): continue
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item:
                    item.setBackground(color)
