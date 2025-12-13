# -*- coding: utf-8 -*-
"""
Splash Screen - หน้า Loading ที่แสดงระหว่าง Import Modules
"""

from PyQt5.QtWidgets import QSplashScreen, QApplication, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
import sys
import os


class LoadingSplashScreen(QSplashScreen):
    """Splash Screen สำหรับแสดงหน้า Loading"""
    
    def __init__(self):
        # สร้าง pixmap สำหรับ splash screen
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(30, 30, 45))  # สีพื้นหลังเข้ม
        
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.SplashScreen |
            Qt.FramelessWindowHint
        )
        
        # โหลด logo
        self.logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logo.png")
        self.logo_pixmap = None
        if os.path.exists(self.logo_path):
            try:
                self.logo_pixmap = QPixmap(self.logo_path)
                if self.logo_pixmap.isNull():
                    self.logo_pixmap = None
            except Exception as e:
                print(f"⚠️ Warning: Could not load logo: {e}")
                self.logo_pixmap = None
        
        # วาดหน้า loading
        self._draw_splash()
        
        # Progress tracking
        self.current_step = 0
        self.total_steps = 0
        self.status_text = "กำลังเริ่มระบบ..."
        
    def _draw_splash(self):
        """วาดหน้า splash screen"""
        # สร้าง pixmap ใหม่เพื่อวาด (ไม่วาดลง self.pixmap() โดยตรง)
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(30, 30, 45))
        
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            

            # วาดพื้นหลัง gradient
            from PyQt5.QtGui import QLinearGradient
            gradient = QLinearGradient(0, 0, 600, 400)
            gradient.setColorAt(0, QColor(30, 30, 45))
            gradient.setColorAt(1, QColor(20, 20, 35))
            painter.fillRect(0, 0, 600, 400, gradient)
            
            # วาด logo (ถ้ามี)
            if hasattr(self, 'logo_pixmap') and self.logo_pixmap and not self.logo_pixmap.isNull():
                # Scale logo ให้เหมาะสม (สูงสุด 80px)
                logo_scaled = self.logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # วาด logo ด้านซ้ายบน
                logo_x = 50
                logo_y = 30
                painter.drawPixmap(logo_x, logo_y, logo_scaled)
            
            # วาด header
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 24, QFont.Bold)
            painter.setFont(font)
            header_x = 150 if (hasattr(self, 'logo_pixmap') and self.logo_pixmap) else 150
            painter.drawText(header_x, 100, "Bottle Detection System")
            
            font_th = QFont("Arial", 18)
            painter.setFont(font_th)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(header_x, 130, "ระบบตรวจจับขวดและ OCR")
            
            # วาด loading bar background
            painter.setPen(QColor(100, 100, 100))
            painter.setBrush(QColor(50, 50, 50))
            painter.drawRoundedRect(50, 250, 500, 30, 5, 5)
            
            # วาด status text
            font_status = QFont("Arial", 12)
            painter.setFont(font_status)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(50, 300, "กำลังโหลดโมดูล...")
            
            # วาด version/copyright
            font_small = QFont("Arial", 10)
            painter.setFont(font_small)
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(50, 380, "Version 1.0 - Loading...")
        finally:
            painter.end()
        
        # อัพเดท splash ด้วย pixmap ใหม่
        if not pixmap.isNull():
            self.setPixmap(pixmap)
        
    def set_progress(self, current, total, status=""):
        """อัพเดท progress"""
        self.current_step = current
        self.total_steps = total
        
        if status:
            self.status_text = status
        else:
            self.status_text = f"กำลังโหลด... ({current}/{total})"
            
        # วาด progress ใหม่
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(30, 30, 45))
        
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            

            # วาดพื้นหลัง gradient
            from PyQt5.QtGui import QLinearGradient
            gradient = QLinearGradient(0, 0, 600, 400)
            gradient.setColorAt(0, QColor(30, 30, 45))
            gradient.setColorAt(1, QColor(20, 20, 35))
            painter.fillRect(0, 0, 600, 400, gradient)
            
            # วาด logo (ถ้ามี)
            if hasattr(self, 'logo_pixmap') and self.logo_pixmap and not self.logo_pixmap.isNull():
                # Scale logo ให้เหมาะสม (สูงสุด 80px)
                logo_scaled = self.logo_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # วาด logo ด้านซ้ายบน
                logo_x = 50
                logo_y = 30
                painter.drawPixmap(logo_x, logo_y, logo_scaled)
            
            # วาด header
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 24, QFont.Bold)
            painter.setFont(font)
            header_x = 150 if (hasattr(self, 'logo_pixmap') and self.logo_pixmap) else 150
            painter.drawText(header_x, 100, "Bottle Detection System")
            
            font_th = QFont("Arial", 18)
            painter.setFont(font_th)
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(header_x, 130, "ระบบตรวจจับขวดและ OCR")
            
            # คำนวณ progress percentage
            if total > 0:
                progress = (current / total) * 100
                progress_width = int((progress / 100) * 500)
            else:
                progress_width = 0
                progress = 0
            
            # วาด loading bar background
            painter.setPen(QColor(100, 100, 100))
            painter.setBrush(QColor(50, 50, 50))
            painter.drawRoundedRect(50, 250, 500, 30, 5, 5)
            
            # วาด loading bar fill (gradient)
            if progress_width > 0:
                bar_gradient = QLinearGradient(50, 250, 50 + progress_width, 280)
                bar_gradient.setColorAt(0, QColor(0, 150, 255))
                bar_gradient.setColorAt(1, QColor(0, 200, 255))
                painter.setBrush(bar_gradient)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(50, 250, progress_width, 30, 5, 5)
            
            # วาด progress text
            painter.setPen(QColor(255, 255, 255))
            font_progress = QFont("Arial", 11, QFont.Bold)
            painter.setFont(font_progress)
            painter.drawText(50, 275, f"{int(progress)}%")
            
            # วาด status text
            font_status = QFont("Arial", 12)
            painter.setFont(font_status)
            painter.setPen(QColor(150, 150, 150))
            status_display = self.status_text[:40]  # จำกัดความยาว
            painter.drawText(50, 300, status_display)
            
            # วาด version/copyright
            font_small = QFont("Arial", 10)
            painter.setFont(font_small)
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(50, 380, "Version 1.0 - Loading...")
        finally:
            painter.end()
        
        # อัพเดท splash - ต้องปิด painter ก่อน
        if pixmap and not pixmap.isNull():
            self.setPixmap(pixmap)
            QApplication.processEvents()  # อัพเดท GUI
        
    def show_message(self, message):
        """แสดงข้อความ"""
        self.set_progress(self.current_step, self.total_steps, message)


def show_splash_screen(app):
    """สร้างและแสดง splash screen"""
    splash = LoadingSplashScreen()
    splash.show()
    
    # Center splash screen
    from PyQt5.QtWidgets import QDesktopWidget
    screen = QDesktopWidget().screenGeometry()
    splash_size = splash.size()
    splash.move((screen.width() - splash_size.width()) // 2, 
                (screen.height() - splash_size.height()) // 2)
    
    app.processEvents()  # แสดง splash screen
    return splash

