#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบการเชื่อมต่อและส่งค่าไปที่ Robot Controller
IP: 192.168.1.162
Address: 0x3101 (12545 ในฐาน 10)
"""

from pymodbus.client import ModbusTcpClient
import time
import sys


class RobotControllerTester:
    def __init__(self, ip_address="192.168.1.162", port=502, address=0x3101):
        """
        Initialize Robot Controller Tester
        
        Args:
            ip_address: IP address ของ robot controller
            port: Modbus TCP port (default: 502)
            address: Modbus address (0x3101 = 12545 ในฐาน 10)
        """
        self.ip_address = ip_address
        self.port = port
        self.address = address  # 0x3101
        self.address_decimal = address  # 12545
        self.client = None
        
    def connect(self):
        """เชื่อมต่อกับ robot controller"""
        try:
            print(f"🔄 กำลังเชื่อมต่อกับ {self.ip_address}:{self.port}...")
            self.client = ModbusTcpClient(host=self.ip_address, port=self.port)
            
            if self.client.connect():
                print(f"✅ เชื่อมต่อสำเร็จ: {self.ip_address}:{self.port}")
                return True
            else:
                print(f"❌ ไม่สามารถเชื่อมต่อได้: {self.ip_address}:{self.port}")
                return False
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
            return False
    
    def disconnect(self):
        """ปิดการเชื่อมต่อ"""
        if self.client:
            self.client.close()
            print("🔌 ปิดการเชื่อมต่อแล้ว")
    
    def write_register(self, value):
        """
        เขียนค่าไปที่ register
        
        Args:
            value: ค่าที่ต้องการเขียน (0-65535)
        """
        try:
            if not self.client or not self.client.is_socket_open():
                print("❌ ยังไม่ได้เชื่อมต่อ กรุณาเรียก connect() ก่อน")
                return False
            
            print(f"🔄 กำลังเขียนค่า {value} ไปที่ address 0x{self.address:04X} ({self.address_decimal})...")
            
            # เขียนค่าไปที่ register
            # unit=1 คือ slave ID (อาจต้องปรับตาม controller)
            result = self.client.write_register(self.address_decimal, value, unit=1)
            
            if result.isError():
                print(f"❌ ไม่สามารถเขียนค่าได้: {result}")
                return False
            else:
                print(f"✅ เขียนค่า {value} สำเร็จที่ address 0x{self.address:04X} ({self.address_decimal})")
                return True
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเขียนค่า: {e}")
            return False
    
    def read_register(self):
        """
        อ่านค่าจาก register
        
        Returns:
            ค่าที่อ่านได้ หรือ None ถ้าเกิดข้อผิดพลาด
        """
        try:
            if not self.client or not self.client.is_socket_open():
                print("❌ ยังไม่ได้เชื่อมต่อ กรุณาเรียก connect() ก่อน")
                return None
            
            print(f"🔄 กำลังอ่านค่าจาก address 0x{self.address:04X} ({self.address_decimal})...")
            
            # อ่านค่าจาก register
            result = self.client.read_holding_registers(self.address_decimal, 1, unit=1)
            
            if result.isError():
                print(f"❌ ไม่สามารถอ่านค่าได้: {result}")
                return None
            else:
                value = result.registers[0]
                print(f"✅ อ่านค่าได้: {value} จาก address 0x{self.address:04X} ({self.address_decimal})")
                return value
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการอ่านค่า: {e}")
            return None
    
    def test_write_read(self, test_value=100):
        """
        ทดสอบการเขียนและอ่านค่า
        
        Args:
            test_value: ค่าที่ต้องการทดสอบ
        """
        print("\n" + "="*60)
        print("🧪 ทดสอบการเขียนและอ่านค่า")
        print("="*60)
        
        # เขียนค่า
        if self.write_register(test_value):
            time.sleep(0.5)  # รอสักครู่
            
            # อ่านค่ากลับมา
            read_value = self.read_register()
            
            if read_value is not None:
                if read_value == test_value:
                    print(f"✅ ทดสอบสำเร็จ: เขียน {test_value} และอ่านได้ {read_value}")
                    return True
                else:
                    print(f"⚠️ ค่าที่อ่านได้ไม่ตรง: เขียน {test_value} แต่อ่านได้ {read_value}")
                    return False
            else:
                print("❌ ไม่สามารถอ่านค่าได้")
                return False
        else:
            print("❌ ไม่สามารถเขียนค่าได้")
            return False


def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    print("="*60)
    print("🤖 Robot Controller Tester")
    print("="*60)
    print(f"IP Address: 192.168.1.162")
    print(f"Address: 0x3101 (12545)")
    print("="*60)
    
    # สร้าง tester
    tester = RobotControllerTester(
        ip_address="192.168.1.162",
        port=502,
        address=0x3101  # 0x3101 = 12545
    )
    
    try:
        # เชื่อมต่อ
        if not tester.connect():
            print("\n❌ ไม่สามารถเชื่อมต่อได้ กรุณาตรวจสอบ:")
            print("   1. IP address ถูกต้องหรือไม่")
            print("   2. Robot controller เปิดอยู่หรือไม่")
            print("   3. Network connection ใช้งานได้หรือไม่")
            return
        
        # เมนูทดสอบ
        while True:
            print("\n" + "="*60)
            print("📋 เมนูทดสอบ")
            print("="*60)
            print("1. เขียนค่าไปที่ register")
            print("2. อ่านค่าจาก register")
            print("3. ทดสอบเขียนและอ่าน (test value = 100)")
            print("4. ทดสอบเขียนและอ่าน (ระบุค่าเอง)")
            print("5. ทดสอบหลายค่า (0, 100, 500, 1000)")
            print("0. ออกจากโปรแกรม")
            print("="*60)
            
            choice = input("เลือกเมนู (0-5): ").strip()
            
            if choice == "0":
                print("👋 ออกจากโปรแกรม")
                break
            elif choice == "1":
                try:
                    value = int(input("กรุณาใส่ค่าที่ต้องการเขียน (0-65535): "))
                    if 0 <= value <= 65535:
                        tester.write_register(value)
                    else:
                        print("❌ ค่าต้องอยู่ระหว่าง 0-65535")
                except ValueError:
                    print("❌ กรุณาใส่ตัวเลข")
            elif choice == "2":
                tester.read_register()
            elif choice == "3":
                tester.test_write_read(100)
            elif choice == "4":
                try:
                    value = int(input("กรุณาใส่ค่าที่ต้องการทดสอบ (0-65535): "))
                    if 0 <= value <= 65535:
                        tester.test_write_read(value)
                    else:
                        print("❌ ค่าต้องอยู่ระหว่าง 0-65535")
                except ValueError:
                    print("❌ กรุณาใส่ตัวเลข")
            elif choice == "5":
                print("\n🧪 ทดสอบหลายค่า...")
                test_values = [0, 100, 500, 1000]
                for val in test_values:
                    print(f"\n--- ทดสอบค่า {val} ---")
                    tester.test_write_read(val)
                    time.sleep(1)  # รอ 1 วินาทีระหว่างการทดสอบ
            else:
                print("❌ กรุณาเลือกเมนู 0-5")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ ถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ปิดการเชื่อมต่อ
        tester.disconnect()


if __name__ == "__main__":
    main()

