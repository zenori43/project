# -*- coding: utf-8 -*-
"""
Login Dialog - Dialog สำหรับเข้าสู่ระบบ Admin
"""

from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QApplication
)
from PyQt5.QtCore import Qt


class LoginDialog(QDialog):
    """Login Dialog for admin access"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('🔐 Login')
        self.setFixedSize(420, 250)
        
        # Center the dialog on the screen
        self.center_dialog()
        self.setStyleSheet("""
            QDialog {
                background-color: #2c3e50;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 14px;
                padding: 10px 8px;
                min-height: 28px;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                min-height: 32px;
                background-color: #34495e;
                color: #ecf0f1;
                border: 2px solid #7f8c8d;
                border-radius: 5px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
            QPushButton {
                padding: 8px 20px;
                font-size: 14px;
                min-height: 32px;
                min-width: 90px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(30, 8, 30, 25)
        
        # Title - positioned at the very top
        layout.addSpacing(0)  # ไม่มี spacing ด้านบน
        title_label = QLabel('Admin Login')
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ecf0f1; padding: 8px 10px; min-height: 38px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addSpacing(15)  # spacing หลัง title
        
        # Username field
        username_layout = QHBoxLayout()
        username_layout.setSpacing(12)
        username_label = QLabel('Username:')
        username_label.setFixedWidth(95)
        username_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        username_layout.addWidget(username_label)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('Enter username')
        self.username_input.setMinimumHeight(34)
        username_layout.addWidget(self.username_input)
        layout.addLayout(username_layout)
        layout.addSpacing(8)  # Add space between fields
        
        # Password field
        password_layout = QHBoxLayout()
        password_layout.setSpacing(12)
        password_label = QLabel('Password:')
        password_label.setFixedWidth(95)
        password_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        password_layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Enter password')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(34)
        password_layout.addWidget(self.password_input)
        layout.addLayout(password_layout)
        layout.addSpacing(15)  # Add space before buttons
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.addStretch()
        
        login_button = QPushButton('Login')
        login_button.setMinimumHeight(34)
        login_button.setMinimumWidth(90)
        login_button.clicked.connect(self.accept)
        button_layout.addWidget(login_button)
        
        cancel_button = QPushButton('Cancel')
        cancel_button.setMinimumHeight(34)
        cancel_button.setMinimumWidth(90)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Set focus to username field
        self.username_input.setFocus()
    
    def center_dialog(self):
        """Center the dialog on the screen, positioned slightly higher"""
        if self.parent():
            # Center relative to parent window, but move up a bit
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2 - 80  # Move up from center
            self.move(x, y)
        else:
            # Center on screen, positioned in upper-middle area
            screen = QApplication.desktop().screenGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 3  # Position at 1/3 from top (upper-middle)
            self.move(x, y)
        
    def get_credentials(self):
        """Return username and password"""
        return self.username_input.text(), self.password_input.text()

