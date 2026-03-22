#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบการส่งค่าไปที่ Robot Controller (แบบง่าย)
Usage:
    python test_robot_simple.py [value]
    
Example:
    python test_robot_simple.py 100
    python test_robot_simple.py 0
"""

from pymodbus.client import ModbusTcpClient
import sys
import time


def test_robot_controller(value=100):
    """
    ทดสอบการส่งค่าไปที่ robot controller
    
    Args:
        value: ค่าที่ต้องการส่ง (0-65535)
    """
    # ตั้งค่า
    IP_ADDRESS = "192.168.1.162"
    PORT = 502
    ADDRESS = 0x3101  # 12545 ในฐาน 10
    UNIT_ID = 2  # Slave ID (อาจต้องปรับตาม controller)
    
    print("="*60)
    print("🤖 Robot Controller Tester (Simple)")
    print("="*60)
    print(f"IP Address: {IP_ADDRESS}")
    print(f"Port: {PORT}")
    print(f"Address: 0x{ADDRESS:04X} ({ADDRESS})")
    print(f"Value: {value}")
    print("="*60)
    
    # เชื่อมต่อ
    print(f"\n🔄 กำลังเชื่อมต่อกับ {IP_ADDRESS}:{PORT}...")
    client = ModbusTcpClient(host=IP_ADDRESS, port=PORT)
    
    try:
        if not client.connect():
            print(f"❌ ไม่สามารถเชื่อมต่อได้: {IP_ADDRESS}:{PORT}")
            print("\n💡 ตรวจสอบ:")
            print("   1. IP address ถูกต้องหรือไม่")
            print("   2. Robot controller เปิดอยู่หรือไม่")
            print("   3. Network connection ใช้งานได้หรือไม่")
            return False
        
        print("✅ เชื่อมต่อสำเร็จ!")
        
        # เขียนค่า
        print(f"\n🔄 กำลังเขียนค่า {value} ไปที่ address 0x{ADDRESS:04X}...")
        result = client.write_register(ADDRESS, value, unit=UNIT_ID)
        
        if result.isError():
            print(f"❌ ไม่สามารถเขียนค่าได้: {result}")
            return False
        
        print(f"✅ เขียนค่า {value} สำเร็จ!")
        
        # อ่านค่ากลับมาเพื่อยืนยัน
        print(f"\n🔄 กำลังอ่านค่ากลับมาเพื่อยืนยัน...")
        time.sleep(0.5)  # รอสักครู่
        
        read_result = client.read_holding_registers(ADDRESS, 1, unit=UNIT_ID)
        
        if read_result.isError():
            print(f"⚠️ ไม่สามารถอ่านค่าได้: {read_result}")
            print("   (แต่การเขียนอาจสำเร็จแล้ว)")
        else:
            read_value = read_result.registers[0]
            print(f"✅ อ่านค่าได้: {read_value}")
            
            if read_value == value:
                print(f"✅ ยืนยันสำเร็จ: ค่าที่เขียนและอ่านตรงกัน ({value})")
            else:
                print(f"⚠️ ค่าที่อ่านได้ไม่ตรง: เขียน {value} แต่อ่านได้ {read_value}")
        
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        client.close()
        print("\n🔌 ปิดการเชื่อมต่อแล้ว")


if __name__ == "__main__":
    import time
    
    # รับค่าจาก command line argument
    if len(sys.argv) > 1:
        try:
            value = int(sys.argv[1])
            if not (0 <= value <= 65535):
                print("❌ ค่าต้องอยู่ระหว่าง 0-65535")
                sys.exit(1)
        except ValueError:
            print("❌ กรุณาใส่ตัวเลข")
            print(f"Usage: {sys.argv[0]} [value]")
            sys.exit(1)
    else:
        # ค่า default
        value = 100
        print("💡 ไม่ได้ระบุค่า ใช้ค่า default = 100")
        print(f"💡 Usage: {sys.argv[0]} [value]")
        print()
    
    # รันทดสอบ
    success = test_robot_controller(value)
    
    if success:
        print("\n✅ ทดสอบเสร็จสิ้น")
        sys.exit(0)
    else:
        print("\n❌ ทดสอบล้มเหลว")
        sys.exit(1)

