# -*- coding: utf-8 -*-
"""
Modbus Simulator - จำลอง PLC/Modbus สำหรับสั่งการโปรแกรม main
รันโปรแกรมนี้ก่อน แล้วตั้งค่าโปรแกรม main ให้เชื่อมต่อมาที่ IP:Port ของ simulator
  - M301 = ปุ่มถ่ายภาพ (ส่งเป็น pulse ON แล้ว OFF)
  - M511 = เริ่มโปรแกรม, M513 = หยุด
  - M401, M402 = สัญญาณตามรอบ
  - M403-M406 = ประเภทขวด
  - D5500 = โหมด (5=Capture only, 7=Auto, 8=Reset)
"""
import sys
import traceback
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QSpinBox, QComboBox, QGridLayout,
    QMessageBox, QLineEdit, QTextEdit, QScrollArea, QFrame,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

# Modbus server (pymodbus 3.x) — 3.12 ใช้ ModbusDeviceContext แทน ModbusSlaveContext
_PYMODBUS_IMPORT_ERROR = None
try:
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
    )
    try:
        from pymodbus.datastore import ModbusDeviceContext as ModbusSlaveContext
    except ImportError:
        from pymodbus.datastore import ModbusSlaveContext
    from pymodbus.server import StartAsyncTcpServer
    try:
        # บางเวอร์ชันของ pymodbus ไม่มีโมดูล device ให้ใช้ identity แบบง่ายๆ แทน
        from pymodbus.device import ModbusDeviceIdentification
    except ImportError:
        ModbusDeviceIdentification = None
    import asyncio
    PYMODBUS_AVAILABLE = True
except ImportError as _e:
    PYMODBUS_AVAILABLE = False
    _PYMODBUS_IMPORT_ERROR = _e

# Coils: 0..999, Holding registers: 0..9999 (main ใช้ unit=1; D5001,D5002 ใช้ unit=2)
COIL_SIZE = 1000
REG_SIZE = 10000
PULSE_MS = 200  # ความกว้าง pulse สำหรับ M301 (ms)

# Modbus function codes (pymodbus 3.12 ใช้ตัวเลข ไม่ใช้ "co"/"hr")
FC_READ_COILS = 1
FC_READ_HOLDING_REGISTERS = 3
FC_WRITE_SINGLE_COIL = 5
FC_WRITE_SINGLE_REGISTER = 6


def make_server_context():
    """สร้าง datastore สำหรับ unit 1 (coils + holding) และ unit 2 (holding สำหรับ D5001, D5002)"""
    coils = ModbusSequentialDataBlock(0, [0] * COIL_SIZE)
    holding = ModbusSequentialDataBlock(0, [0] * REG_SIZE)
    store1 = ModbusSlaveContext(co=coils, hr=holding)
    holding2 = ModbusSequentialDataBlock(0, [0] * REG_SIZE)
    store2 = ModbusSlaveContext(hr=holding2)
    try:
        ctx = ModbusServerContext(devices={1: store1, 2: store2}, single=False)
    except TypeError:
        ctx = ModbusServerContext(slaves={1: store1, 2: store2}, single=False)
    return ctx, store1, store2


def set_coil(store, address, value):
    """ตั้งค่า coil (unit 1)"""
    if store is None:
        return False
    try:
        store.setValues(FC_WRITE_SINGLE_COIL, address, [bool(value)])
        return True
    except Exception:
        try:
            store.setValues("co", address, [bool(value)])
            return True
        except Exception:
            return False


def set_register(store, address, value):
    """ตั้งค่า holding register"""
    if store is None:
        return False
    try:
        store.setValues(FC_WRITE_SINGLE_REGISTER, address, [int(value) & 0xFFFF])
        return True
    except Exception:
        try:
            store.setValues("hr", address, [int(value) & 0xFFFF])
            return True
        except Exception:
            return False


def get_register(store, address):
    """อ่านค่า holding register"""
    if store is None:
        return None
    try:
        vals = store.getValues(FC_READ_HOLDING_REGISTERS, address, count=1)
        return vals[0] if vals else None
    except Exception:
        try:
            vals = store.getValues("hr", address, count=1)
            return vals[0] if vals else None
        except Exception:
            return None


def get_coil(store, address):
    """อ่านค่า coil (ที่ main อาจ ON/OFF ไว้)"""
    if store is None:
        return None
    try:
        vals = store.getValues(FC_READ_COILS, address, count=1)
        return bool(vals[0]) if vals else None
    except Exception:
        try:
            vals = store.getValues("co", address, count=1)
            return bool(vals[0]) if vals else None
        except Exception:
            return None


class ModbusServerThread(threading.Thread):
    """รัน Modbus TCP server ใน thread แยก (ใช้ asyncio)"""
    def __init__(self, context, host="0.0.0.0", port=5502):
        super().__init__(daemon=True)
        self.context = context
        self.host = host
        self.port = port
        self._loop = None
        self._server = None
        self._error_msg = None  # เก็บข้อความ error ให้ GUI อ่าน

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            kwargs = {"address": (self.host, self.port)}
            if ModbusDeviceIdentification is not None:
                identity = ModbusDeviceIdentification()
                identity.VendorName = "ModbusSimulator"
                identity.ProductName = "Bottle Detection PLC Sim"
                kwargs["identity"] = identity
            self._loop.run_until_complete(
                StartAsyncTcpServer(
                    self.context,
                    **kwargs,
                )
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._error_msg = f"{e}\n{traceback.format_exc()}"
        finally:
            if self._loop:
                try:
                    self._loop.close()
                except Exception:
                    pass

    def stop(self):
        if self._loop and self._server:
            try:
                self._loop.call_soon_threadsafe(self._server.shutdown)
            except Exception:
                pass


class ModbusSimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.context = None
        self.store1 = None
        self.store2 = None
        self.server_thread = None
        self.log_lines = []

        self.setWindowTitle("Modbus Simulator - จำลอง PLC สั่งการโปรแกรม Main")
        self.setMinimumSize(520, 620)
        self.resize(560, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- หมายเหตุ ---
        note = QLabel(
            "ใช้โปรแกรมนี้รันก่อน แล้วตั้งโปรแกรม main: config/settings.py → MODBUS_IP = \"127.0.0.1\", MODBUS_PORT = 5502"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #f39c12; background: #2c3e50; padding: 8px; border-radius: 4px;")
        layout.addWidget(note)

        # --- Server ---
        server_group = QGroupBox("เซิร์ฟเวอร์ Modbus TCP")
        server_layout = QHBoxLayout(server_group)
        self.host_edit = QLineEdit("0.0.0.0")
        self.host_edit.setPlaceholderText("0.0.0.0 หรือ 127.0.0.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5502)
        self.port_spin.setToolTip("ใช้ 5502 เพื่อไม่ต้องรัน as admin (port 502 ต้องสิทธิ์)")
        self.btn_start = QPushButton("เริ่มเซิร์ฟเวอร์")
        self.btn_stop = QPushButton("หยุดเซิร์ฟเวอร์")
        self.btn_stop.setEnabled(False)
        self.status_label = QLabel("สถานะ: ไม่ได้รัน")
        self.status_label.setStyleSheet("color: #95a5a6;")
        server_layout.addWidget(QLabel("Host:"))
        server_layout.addWidget(self.host_edit)
        server_layout.addWidget(QLabel("Port:"))
        server_layout.addWidget(self.port_spin)
        server_layout.addWidget(self.btn_start)
        server_layout.addWidget(self.btn_stop)
        server_layout.addWidget(self.status_label)
        layout.addWidget(server_group)

        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop.clicked.connect(self.stop_server)

        # --- ปุ่มหลัก ---
        main_group = QGroupBox("สัญญาณหลัก")
        main_layout = QGridLayout(main_group)

        self._add_pulse_btn(main_layout, 0, 0, "M301 ถ่าย", 301, "ส่งสัญญาณถ่ายภาพ (pulse)")
        self._add_toggle_btn(main_layout, 0, 1, "M511 เริ่ม", 511, "เริ่มโปรแกรม")
        self._add_toggle_btn(main_layout, 0, 2, "M513 หยุด", 513, "หยุดโปรแกรม")
        self._add_toggle_btn(main_layout, 1, 0, "M401", 401, "M401 ON (reset M600 / แสดงผล)")
        self._add_toggle_btn(main_layout, 1, 1, "M402", 402, "M402 (ถ่ายซ้ำ angle3)")
        self._add_toggle_btn(main_layout, 1, 2, "M512", 512, "M512 (พร้อมรับ M513)")
        layout.addWidget(main_group)

        # --- ประเภทขวด (M403-M406) ---
        bottle_group = QGroupBox("ประเภทขวด (M403-M406)")
        bottle_layout = QHBoxLayout(bottle_group)
        for i, (addr, name) in enumerate([(403, "M403"), (404, "M404"), (405, "M405"), (406, "M406")]):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setProperty("coil", addr)
            btn.clicked.connect(self._on_bottle_toggle)
            bottle_layout.addWidget(btn)
            setattr(self, f"btn_m{addr}", btn)
        layout.addWidget(bottle_group)

        # --- โหมด D5500 ---
        mode_group = QGroupBox("โหมด D5500")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("5 - Capture only (ถ่ายอย่างเดียว)", 5)
        self.mode_combo.addItem("7 - Auto", 7)
        self.mode_combo.addItem("8 - ID 8 reset modbus", 8)
        self.btn_apply_mode = QPushButton("ตั้ง D5500")
        self.btn_apply_mode.clicked.connect(self.apply_d5500)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(self.btn_apply_mode)
        layout.addWidget(mode_group)

        # --- D5001 / D5002 (Unit 2 - ส่งเข้า Main) ---
        d50_group = QGroupBox("D5001 / D5002 (Unit 2 - ส่งเข้า Main)")
        d50_layout = QHBoxLayout(d50_group)
        d50_layout.addWidget(QLabel("D5001:"))
        self.d5001_label = QLabel("0 (คงที่)")
        self.d5001_label.setToolTip("ส่ง 0 เข้า Main เสมอ (Unit 2)")
        d50_layout.addWidget(self.d5001_label)
        d50_layout.addWidget(QLabel("D5002:"))
        self.d5002_combo = QComboBox()
        for i in range(5):
            self.d5002_combo.addItem(str(i), i)
        self.d5002_combo.setCurrentIndex(0)  # default 0
        self.d5002_combo.setToolTip("Main อ่านจาก Unit 2 - เปลี่ยนได้ 0,1,2,3,4 (default 0)")
        self.d5002_combo.currentIndexChanged.connect(self._apply_d5001_d5002)
        d50_layout.addWidget(self.d5002_combo)
        layout.addWidget(d50_group)

        # --- สัญญาณเพิ่ม (Start/Stop ระบบ, M700/M701) ---
        extra_group = QGroupBox("Start/Stop ระบบ")
        extra_layout = QHBoxLayout(extra_group)
        self._add_toggle_btn(extra_layout, 0, 0, "M700 เริ่ม", 700, "เริ่มระบบ", as_grid=False)
        self._add_toggle_btn(extra_layout, 0, 1, "M701 หยุด", 701, "หยุดระบบ", as_grid=False)
        layout.addWidget(extra_group)

        # --- Register + Coil ที่ main ส่งมา (จำลองอ่านได้ว่า main เขียน/ON อะไร) ---
        main_sent_group = QGroupBox("📥 ที่ Main ส่งมา (Registers + Coils)")
        main_sent_layout = QVBoxLayout(main_sent_group)
        reg_row = QHBoxLayout()
        self.d7009_label = QLabel("D7009: -")
        self.d5500_label = QLabel("D5500: -")
        self.d9006_label = QLabel("D9006: -")
        self.refresh_reg_btn = QPushButton("รีเฟรช")
        self.refresh_reg_btn.clicked.connect(self.refresh_register_display)
        reg_row.addWidget(self.d7009_label)
        reg_row.addWidget(self.d5500_label)
        reg_row.addWidget(self.d9006_label)
        reg_row.addWidget(self.refresh_reg_btn)
        main_sent_layout.addLayout(reg_row)
        self.main_sent_coils_text = QLabel("Coils: M503 M600 M511 M512 M513 M401 M402 M403-406 M700 M701 ...")
        self.main_sent_coils_text.setWordWrap(True)
        self.main_sent_coils_text.setStyleSheet("font-family: Consolas; font-size: 10px; color: #2c3e50;")
        self.main_sent_coils_text.setToolTip("สถานะ coil ที่ main ON/OFF (อัปเดตทุก 1 วินาที)")
        main_sent_layout.addWidget(self.main_sent_coils_text)
        layout.addWidget(main_sent_group)

        # --- Log ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        self._reg_timer = QTimer(self)
        self._reg_timer.timeout.connect(self.refresh_register_display)
        self._reg_timer.start(1000)

        # แสดง Python ที่ใช้รัน (ช่วยเช็กว่าตรงกับที่ติดตั้ง pymodbus หรือไม่)
        try:
            _py_exe = sys.executable or "python"
            _py_ver = getattr(sys, "version", "?")[:6].strip()
            self._log(f"Python: {_py_exe} (เวอร์ชัน {_py_ver})")
        except Exception:
            pass
        if not PYMODBUS_AVAILABLE:
            self._log("⚠️ ไม่พบ pymodbus ใน Python ตัวนี้")
            self._log("   ติดตั้ง: py -3.11 -m pip install \"pymodbus>=3.0.0\"")
            self._log("   แล้วรัน simulator ด้วย: py -3.11 tools\\modbus_simulator.py")
            if _PYMODBUS_IMPORT_ERROR:
                self._log(f"   Error: {_PYMODBUS_IMPORT_ERROR}")
            self.btn_start.setEnabled(False)
        else:
            self._log("พร้อมรันเซิร์ฟเวอร์ Modbus (ปุ่ม M301 = ถ่าย)")

    def _log(self, msg):
        self.log_lines.append(msg)
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)
        self.log_text.setText("\n".join(self.log_lines[-50:]))
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _add_pulse_btn(self, layout, row, col, label, coil_addr, tooltip, as_grid=True):
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.setProperty("coil", coil_addr)
        btn.setProperty("pulse", True)
        btn.clicked.connect(self._on_coil_click)
        if as_grid:
            layout.addWidget(btn, row, col)
        else:
            layout.addWidget(btn)

    def _add_toggle_btn(self, layout, row, col, label, coil_addr, tooltip, as_grid=True):
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setProperty("coil", coil_addr)
        btn.setProperty("pulse", False)
        btn.clicked.connect(self._on_coil_click)
        if as_grid:
            layout.addWidget(btn, row, col)
        else:
            layout.addWidget(btn)

    def _on_coil_click(self):
        sender = self.sender()
        if not isinstance(sender, QPushButton):
            return
        coil = sender.property("coil")
        is_pulse = sender.property("pulse")
        if coil is None:
            return
        if is_pulse:
            self._send_pulse(coil)
        else:
            value = sender.isChecked()
            self._set_coil(coil, value)

    def _on_bottle_toggle(self):
        sender = self.sender()
        if not isinstance(sender, QPushButton):
            return
        coil = sender.property("coil")
        if coil is None:
            return
        # เปิดแค่ตัวที่กด ปิดตัวอื่นในกลุ่ม 403-406
        for addr in (403, 404, 405, 406):
            self._set_coil(addr, addr == coil and sender.isChecked())

    def _set_coil(self, address, value):
        if self.store1 is None:
            self._log("⚠️ เซิร์ฟเวอร์ยังไม่รัน")
            return
        set_coil(self.store1, address, value)
        self._log(f"Coil M{address} = {1 if value else 0}")

    def _send_pulse(self, address):
        if self.store1 is None:
            self._log("⚠️ เซิร์ฟเวอร์ยังไม่รัน")
            return
        set_coil(self.store1, address, True)
        self._log(f"M{address} ON (pulse)")
        QTimer.singleShot(PULSE_MS, lambda: self._pulse_off(address))

    def _pulse_off(self, address):
        if self.store1:
            set_coil(self.store1, address, False)
            self._log(f"M{address} OFF")

    def apply_d5500(self):
        if self.store1 is None:
            self._log("⚠️ เซิร์ฟเวอร์ยังไม่รัน")
            return
        val = self.mode_combo.currentData()
        set_register(self.store1, 5500, val)
        self._log(f"D5500 = {val}")

    def _apply_d5001_d5002(self):
        """ตั้ง D5001=0 (คงที่) และ D5002 (0-4) บน Unit 2 ให้ main อ่าน"""
        if self.store2 is None:
            return
        set_register(self.store2, 5001, 0)  # D5001 ส่ง 0 เข้า main เสมอ
        d5002_val = self.d5002_combo.currentData()
        set_register(self.store2, 5002, d5002_val)
        self._log(f"D5001 = 0, D5002 = {d5002_val} (Unit 2)")

    def refresh_register_display(self):
        if self.store1 is None:
            return
        d7 = get_register(self.store1, 7009)
        d55 = get_register(self.store1, 5500)
        d90 = get_register(self.store1, 9006)
        self.d7009_label.setText(f"D7009: {d7 if d7 is not None else '-'}")
        self.d5500_label.setText(f"D5500: {d55 if d55 is not None else '-'}")
        self.d9006_label.setText(f"D9006: {d90 if d90 is not None else '-'}")
        # Coils ที่ main อาจ ON/OFF
        coil_addrs = [503, 600, 511, 512, 513, 401, 402, 403, 404, 405, 406, 700, 701]
        parts = []
        for addr in coil_addrs:
            v = get_coil(self.store1, addr)
            if v is None:
                parts.append(f"M{addr}:?")
            else:
                parts.append(f"M{addr}:{'ON' if v else 'OFF'}")
        self.main_sent_coils_text.setText("Coils (จาก Main): " + "  ".join(parts))

    def start_server(self):
        if not PYMODBUS_AVAILABLE:
            self._log("⚠️ ไม่พบ pymodbus - ติดตั้ง: pip install pymodbus>=3.0.0")
            return
        host = self.host_edit.text().strip() or "0.0.0.0"
        port = self.port_spin.value()
        self._log(f"🔄 กำลังเริ่มเซิร์ฟเวอร์ Modbus ที่ {host}:{port} ...")
        QApplication.processEvents()
        try:
            self.context, self.store1, self.store2 = make_server_context()
            self._apply_d5001_d5002()  # D5001=0 (คงที่), D5002=0 (default) บน Unit 2 ให้ main อ่าน
            self.server_thread = ModbusServerThread(self.context, host=host, port=port)
            self.server_thread.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_label.setText(f"กำลังเริ่ม... {host}:{port}")
            self.status_label.setStyleSheet("color: #f39c12;")
            # ตรวจสอบหลัง 1.2 วินาที ว่า thread ยังรันอยู่หรือ error
            QTimer.singleShot(1200, self._check_server_started)
        except Exception as e:
            err = traceback.format_exc()
            self._log(f"❌ เริ่มเซิร์ฟเวอร์ไม่สำเร็จ: {e}")
            self._log(err)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.status_label.setText("สถานะ: ไม่ได้รัน")
            self.status_label.setStyleSheet("color: #95a5a6;")
            QMessageBox.warning(self, "ข้อผิดพลาด", f"{e}\n\nดู Log ด้านล่างสำหรับรายละเอียด")

    def _check_server_started(self):
        """ตรวจหลังเริ่ม thread ว่าเซิร์ฟเวอร์ขึ้นได้หรือ error"""
        if self.server_thread is None:
            return
        if not self.server_thread.is_alive():
            # thread หยุดไปแล้ว = น่าจะ error ตอน start
            err = getattr(self.server_thread, "_error_msg", None) or "เซิร์ฟเวอร์หยุดโดยไม่ทราบสาเหตุ"
            self._log("❌ เซิร์ฟเวอร์เริ่มไม่สำเร็จ:")
            for line in err.strip().split("\n"):
                self._log(f"   {line}")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.status_label.setText("สถานะ: เริ่มไม่สำเร็จ")
            self.status_label.setStyleSheet("color: #e74c3c;")
            self.context = None
            self.store1 = None
            self.store2 = None
            QMessageBox.warning(
                self,
                "เซิร์ฟเวอร์เริ่มไม่สำเร็จ",
                "ดู Log ด้านล่าง\n\nสาเหตุที่พบบ่อย: พอร์ตถูกใช้อยู่แล้ว (ลองเปลี่ยน Port เป็น 5503 หรือ 5020)",
            )
            return
        # thread ยังรันอยู่ = server น่าจะ bind port ได้แล้ว
        host = self.host_edit.text().strip() or "0.0.0.0"
        port = self.port_spin.value()
        self.status_label.setText(f"รันอยู่ {host}:{port}")
        self.status_label.setStyleSheet("color: #2ecc71;")
        self._log(f"✅ เริ่มเซิร์ฟเวอร์ Modbus ที่ {host}:{port} สำเร็จ")

    def stop_server(self):
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("สถานะ: ไม่ได้รัน")
        self.status_label.setStyleSheet("color: #95a5a6;")
        self._log("หยุดเซิร์ฟเวอร์แล้ว")

    def closeEvent(self, event):
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ModbusSimulatorWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
