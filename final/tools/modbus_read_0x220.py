#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modbus อ่านค่า Holding Register 0x0220 (Robot / MS Controller)
เชื่อมต่อ IP 192.168.1.162 (EtherNet → ใช้ Modbus TCP เท่านั้น)

คู่มือ Controller:
  - โหมด: ASCII, RTU (RS-232/RS-485 หรือ USB-Serial), TCP (EtherNet เท่านั้น)
  - P3-02 = ตั้ง protocol สำหรับ RS-232/RS-485 (ASCII/RTU)
  - ฟังก์ชันที่รองรับ: 03H (อ่าน), 06H (เขียน 1 เรจิสเตอร์), 10H (เขียนหลายเรจิสเตอร์)
  - สคริปต์นี้ใช้ 03H (Read Holding Registers) ผ่าน TCP

Usage:
    cd final
    python3 tools/modbus_read_0x220.py              # อ่าน 0x0220
    python3 tools/modbus_read_0x220.py 0x3101       # อ่าน 0x3101 (แอดเดรสที่โปรเจกต์ใช้กับ robot)
"""

from pymodbus.client import ModbusTcpClient
import sys
import os
import time

# เพิ่ม path ให้ import ได้ถ้ารันจาก final/
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

MODBUS_IP = "192.168.1.162"
MODBUS_PORT = 502
DEFAULT_ADDRESS = 0x0220   # 544 decimal (Holding Register)
UNIT_ID = 2                # Robot controller ใช้ slave/unit = 2 (เหมือน test_robot_simple.py)
COUNT = 1                  # อ่านกี่เรจิสเตอร์
# แอดเดรสที่ลองอ่านตามลำดับ (ถ้า 0x0220 ไม่ได้) — 0x3101 = ที่ test_robot_simple ใช้
FALLBACK_ADDRESSES = [0x3101, 0x0200, 0x0220]


def _parse_address(s):
    s = s.strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)


def _print_tips():
    print("\n💡 Connection reset ทุกแอดเดรส/unit อาจเพราะ:")
    print("   - Controller ยังไม่เปิด Modbus TCP หรือใช้พอร์ตอื่น (ไม่ใช่ 502)")
    print("   - ต้องตั้งค่าในเมนู robot ให้ใช้โหมด Modbus TCP over EtherNet")
    print("   - ลองอ่านแอดเดรสอื่น: python3 tools/modbus_read_0x220.py 0x3101")


def main():
    # รองรับอาร์กิวเมนต์: python3 modbus_read_0x220.py [0x0220|0x3101|544]
    if len(sys.argv) > 1:
        try:
            addr = _parse_address(sys.argv[1])
        except ValueError:
            print("ใช้แอดเดรสเป็นเลขฐานสิบหรือ hex เช่น 0x0220 หรือ 544")
            return 1
        addresses_to_try = [addr]
    else:
        addresses_to_try = [DEFAULT_ADDRESS] + [a for a in FALLBACK_ADDRESSES if a != DEFAULT_ADDRESS]
        addresses_to_try = list(dict.fromkeys(addresses_to_try))

    print("=" * 50)
    print("Modbus TCP อ่าน Holding Register (Robot 192.168.1.162)")
    print("=" * 50)
    print("IP:     {}".format(MODBUS_IP))
    print("Port:   {}".format(MODBUS_PORT))
    print("แอดเดรสที่จะลอง: " + ", ".join("0x{:04X}".format(a) for a in addresses_to_try))
    print("Unit:   {} (และลอง 1, 255 ถ้า reset)".format(UNIT_ID))
    print("=" * 50)

    client = ModbusTcpClient(host=MODBUS_IP, port=MODBUS_PORT)

    try:
        print("\nกำลังเชื่อมต่อ...")
        if not client.connect():
            print("ไม่สามารถเชื่อมต่อได้")
            print("ตรวจสอบ: IP, สายเน็ต, robot เปิดอยู่")
            return 1

        print("เชื่อมต่อสำเร็จ")
        time.sleep(0.2)

        unit_ids_to_try = [UNIT_ID, 1, 2, 255]
        unit_ids_to_try = list(dict.fromkeys(unit_ids_to_try))

        for addr in addresses_to_try:
            for uid in unit_ids_to_try:
                try:
                    if addr != addresses_to_try[0] or uid != unit_ids_to_try[0]:
                        client.close()
                        time.sleep(0.3)
                        if not client.connect():
                            continue
                        time.sleep(0.2)
                    print("กำลังอ่าน Holding 0x{:04X} (unit={})...".format(addr, uid))
                    result = client.read_holding_registers(addr, COUNT, unit=uid)
                except Exception as e:
                    print("  ไม่สำเร็จ: {}".format(e))
                    continue
                if result.isError():
                    print("  ไม่สำเร็จ: {}".format(result))
                    continue
                value = result.registers[0]
                print("ค่าที่อ่านได้ (0x{:04X}, unit={}): {} (0x{:04X})".format(addr, uid, value, value))
                return 0

        _print_tips()
        return 1

    except Exception as e:
        print("เกิดข้อผิดพลาด:", e)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
