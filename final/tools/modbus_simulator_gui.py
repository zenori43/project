import sys

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QLineEdit,
    QGroupBox,
)
from PyQt5.QtCore import Qt

from pymodbus.client import ModbusTcpClient


class ModbusSimulatorGUI(QWidget):
    """
    GUI จำลองการส่งสัญญาณ Modbus ไปยัง PLC/โปรแกรมหลัก

    ใช้สำหรับ:
    - ยิง M301 (Trigger ถ่ายภาพ)
    - ยิง M401 (Reset หลังส่งผล / ใช้ร่วมกับ M600)
    - ยิง M403–M406 (เลือกประเภทขวด)
    - เขียนค่าไปที่รีจิสเตอร์ (เช่น D7009) แบบ manual
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modbus Simulator GUI")
        self.resize(500, 400)

        self.client = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Connection group
        conn_group = QGroupBox("การเชื่อมต่อ Modbus TCP")
        conn_layout = QHBoxLayout(conn_group)

        self.ip_input = QLineEdit("192.168.1.5")
        self.ip_input.setPlaceholderText("Modbus IP")
        self.port_input = QLineEdit("502")
        self.port_input.setPlaceholderText("Port")

        self.connect_btn = QPushButton("เชื่อมต่อ")
        self.connect_btn.clicked.connect(self.on_connect_clicked)

        self.conn_status_label = QLabel("ยังไม่ได้เชื่อมต่อ")
        self.conn_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        conn_layout.addWidget(QLabel("IP:"))
        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.connect_btn)

        main_layout.addWidget(conn_group)
        main_layout.addWidget(self.conn_status_label)

        # Coils group (Mxxx)
        coils_group = QGroupBox("สัญญาณคอยล์ (Mxxx)")
        coils_layout = QGridLayout(coils_group)
        coils_layout.setSpacing(8)

        # แถว 1: Trigger / Reset / Program
        self._add_coil_button_row(
            coils_layout,
            row=0,
            items=[
                ("M301 Trigger", 301),
                ("M401 Reset", 401),
                ("M511 Start Program", 511),
                ("M513 Stop Program", 513),
            ],
        )

        # แถว 2: Taste result (M403–M406)
        self._add_coil_button_row(
            coils_layout,
            row=1,
            items=[
                ("M403 รสดั้งเดิม", 403),
                ("M404 น้ำตาลน้อย 2%", 404),
                ("M405 ผสมแมงลัก", 405),
                ("M406 NG เต็ม", 406),
            ],
        )

        # แถว 3: Bottle type result (M100/M110/M120/M140 via D7009)
        bottle_group = QGroupBox("ผลลัพธ์ขวด/ฝา (D7009)")
        bottle_layout = QHBoxLayout(bottle_group)

        self.m100_btn = QPushButton("M100 (ดั้งเดิม) → D7009=10")
        self.m110_btn = QPushButton("M110 (2%) → D7009=20")
        self.m120_btn = QPushButton("M120 (แมงลัก) → D7009=30")
        self.m140_btn = QPushButton("M140 (ฝา NG) → D7009=50")

        self.m100_btn.clicked.connect(lambda: self.write_register_value(7009, 10))
        self.m110_btn.clicked.connect(lambda: self.write_register_value(7009, 20))
        self.m120_btn.clicked.connect(lambda: self.write_register_value(7009, 30))
        self.m140_btn.clicked.connect(lambda: self.write_register_value(7009, 50))

        bottle_layout.addWidget(self.m100_btn)
        bottle_layout.addWidget(self.m110_btn)
        bottle_layout.addWidget(self.m120_btn)
        bottle_layout.addWidget(self.m140_btn)

        main_layout.addWidget(coils_group)
        main_layout.addWidget(bottle_group)

        # Manual register write
        reg_group = QGroupBox("เขียนค่า Register (Dxxxx) แบบ manual")
        reg_layout = QHBoxLayout(reg_group)

        self.reg_addr_input = QLineEdit("7009")
        self.reg_addr_input.setPlaceholderText("เช่น 7009")
        self.reg_value_input = QLineEdit("")
        self.reg_value_input.setPlaceholderText("ค่า (integer)")

        self.reg_write_btn = QPushButton("เขียนค่า")
        self.reg_write_btn.clicked.connect(self.on_write_register_clicked)

        reg_layout.addWidget(QLabel("D"))
        reg_layout.addWidget(self.reg_addr_input)
        reg_layout.addWidget(QLabel("="))
        reg_layout.addWidget(self.reg_value_input)
        reg_layout.addWidget(self.reg_write_btn)

        main_layout.addWidget(reg_group)

        # Log label (สั้นๆ)
        self.log_label = QLabel("")
        self.log_label.setStyleSheet("color: #555;")
        main_layout.addWidget(self.log_label)

        main_layout.addStretch()

    def _add_coil_button_row(self, layout, row, items):
        """
        สร้างแถวของปุ่มสำหรับ ON/RESET คอยล์

        items: list of (label, coil_address)
        """

        for col, (label, addr) in enumerate(items):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, a=addr, l=label: self.pulse_coil(a, l))
            layout.addWidget(btn, row, col)

    # ---------- Modbus helpers ----------
    def ensure_client(self):
        if self.client and self.client.is_socket_open():
            return True
        return False

    def on_connect_clicked(self):
        ip = self.ip_input.text().strip()
        port_text = self.port_input.text().strip()
        try:
            port = int(port_text)
        except ValueError:
            self.log("Port ต้องเป็นตัวเลข")
            return

        # ปิด client เดิมถ้ามี
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass

        self.client = ModbusTcpClient(ip, port=port)
        if self.client.connect():
            self.conn_status_label.setText(f"เชื่อมต่อแล้ว: {ip}:{port}")
            self.log("เชื่อมต่อ Modbus สำเร็จ")
        else:
            self.conn_status_label.setText("เชื่อมต่อไม่สำเร็จ")
            self.log("❌ เชื่อมต่อ Modbus ไม่ได้")

    def pulse_coil(self, address, label):
        """
        ON coil สั้นๆ แล้ว RESET กลับ 0 (เหมือนกดปุ่ม)
        """
        if not self.ensure_client():
            self.log("ยังไม่ได้เชื่อมต่อ Modbus")
            return

        try:
            # ON
            result_on = self.client.write_coil(address, True, unit=1)
            if result_on.isError():
                self.log(f"❌ เขียน M{address}=1 ไม่สำเร็จ")
                return

            # RESET (OFF) ทันที (ดีพอสำหรับ simulator)
            result_off = self.client.write_coil(address, False, unit=1)
            if result_off.isError():
                self.log(f"⚠️ M{address} ON แล้วแต่ RESET ไม่สำเร็จ")
                return

            self.log(f"✅ Pulse {label} (M{address}) เรียบร้อย")
        except Exception as e:
            self.log(f"❌ Error เขียน M{address}: {e}")

    def write_register_value(self, d_addr, value):
        """
        เขียนค่าไปที่ D-register (holding register)
        """
        if not self.ensure_client():
            self.log("ยังไม่ได้เชื่อมต่อ Modbus")
            return

        try:
            result = self.client.write_register(d_addr, int(value), unit=1)
            if result.isError():
                self.log(f"❌ เขียน D{d_addr} = {value} ไม่สำเร็จ")
                return
            self.log(f"✅ เขียน D{d_addr} = {value} เรียบร้อย")
        except Exception as e:
            self.log(f"❌ Error เขียน D{d_addr}: {e}")

    def on_write_register_clicked(self):
        addr_text = self.reg_addr_input.text().strip()
        val_text = self.reg_value_input.text().strip()

        if not addr_text or not val_text:
            self.log("กรุณากรอกทั้ง address และ value")
            return

        try:
            addr = int(addr_text)
            val = int(val_text)
        except ValueError:
            self.log("address และ value ต้องเป็นตัวเลข")
            return

        self.write_register_value(addr, val)

    # ---------- misc ----------
    def log(self, msg):
        self.log_label.setText(msg)
        print(msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ModbusSimulatorGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

