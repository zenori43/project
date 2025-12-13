# -*- coding: utf-8 -*-
"""
Debug Stream - Custom stream class to capture print statements and display in debug panel
"""

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
        """Override write to capture print statements"""
        # Write to original stream (console)
        if self.original_stream:
            self.original_stream.write(text)
            self.original_stream.flush()
        
        # Add to buffer (handle partial lines)
        self.buffer += text
        
        # Process complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():  # Only process non-empty lines
                # Emit signal to update GUI (thread-safe)
                self.message_written.emit(line)
    
    def flush(self):
        """Flush the stream"""
        if self.original_stream:
            self.original_stream.flush()
    
    def isatty(self):
        """Check if stream is a TTY"""
        if self.original_stream:
            return self.original_stream.isatty()
        return False

