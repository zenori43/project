# -*- coding: utf-8 -*-
"""
Debug Stream - Custom stream class to capture print statements and display in debug panel
เขียนลง original stream (รวม app.log เมื่อรันผ่าน run.sh) เป็นบรรทัดพร้อม [HH:MM:SS]
"""

from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal


class DebugStream(QObject):
    """Custom stream class to capture print statements and display in debug panel"""
    message_written = pyqtSignal(str)
    
    def __init__(self, original_stream, gui_instance=None):
        super().__init__()
        self.original_stream = original_stream
        self.gui_instance = gui_instance
        self.buffer = ""
    
    def write(self, text):
        """Override write to capture print statements; เขียนลง original (app.log) พร้อมเวลา [HH:MM:SS]"""
        # Add to buffer (handle partial lines)
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if self.original_stream:
                stamped = datetime.now().strftime("[%H:%M:%S] ") + line
                self.original_stream.write(stamped + "\n")
                self.original_stream.flush()
            if line.strip():  # Only process non-empty lines for GUI
                self.message_written.emit(line)
    
    def flush(self):
        """Flush the stream; เขียน buffer ที่เหลือ (บรรทัดไม่มี \\n ตอนจบ) พร้อมเวลา"""
        if self.original_stream and self.buffer:
            stamped = datetime.now().strftime("[%H:%M:%S] ") + self.buffer
            self.original_stream.write(stamped)
            self.original_stream.flush()
            self.buffer = ""
        elif self.original_stream:
            self.original_stream.flush()
    
    def isatty(self):
        """Check if stream is a TTY"""
        if self.original_stream:
            return self.original_stream.isatty()
        return False

