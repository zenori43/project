# -*- coding: utf-8 -*-
"""
Logic การหมุนภาพฝาตามกรอบ CRAFT ให้สี่เหลี่ยมตรงแนวนอน
ใช้ร่วมกันได้ทั้งโปรแกรมหลัก (business_logic) และ test_cap_rotation_ui
"""

import cv2
import numpy as np


def compute_longest_edge_angle_and_points(box):
    """
    รับ box (4x2) แล้ว:
    - หาขอบที่ยาวที่สุด (4 ขอบรอบสี่เหลี่ยม)
    - บังคับทิศ ซ้าย -> ขวา
    - คืนค่า (angle_deg, P1, P2)
    """
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)

    best_pair = None
    best_len_sq = -1.0

    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length_sq = dx * dx + dy * dy
        if length_sq > best_len_sq:
            best_len_sq = length_sq
            best_pair = (p1, p2)

    if best_pair is None:
        return 0.0, (0, 0), (0, 0)

    p1, p2 = best_pair

    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = float(np.degrees(np.arctan2(-dy, dx)))

    P1 = (int(round(p1[0])), int(round(p1[1])))
    P2 = (int(round(p2[0])), int(round(p2[1])))
    return angle, P1, P2


def get_craft_box_from_result(craft_result):
    """
    ดึงกรอบ CRAFT ชิ้นแรกจาก craft_result (final_polys ก่อน แล้วค่อย text_polys).
    คืนค่า np.ndarray shape (4,2) หรือ None
    """
    if craft_result is None:
        return None
    box = None
    fp = craft_result.get("final_polys")
    if fp is not None:
        if isinstance(fp, np.ndarray) and fp.size > 0:
            box = np.array(fp[0], dtype=np.float32).reshape(-1, 2)
        elif isinstance(fp, (list, tuple)) and len(fp) > 0:
            box = np.array(fp[0], dtype=np.float32).reshape(-1, 2)
    if box is None:
        _raw = craft_result.get("text_polys") or craft_result.get("text_boxes")
        if _raw is not None:
            if isinstance(_raw, np.ndarray) and _raw.size >= 8:
                box = np.array(_raw[0], dtype=np.float32).reshape(-1, 2)
            elif isinstance(_raw, (list, tuple)) and len(_raw) > 0:
                box = np.array(_raw[0], dtype=np.float32).reshape(-1, 2)
    return box if (box is not None and len(box) >= 4) else None


def apply_craft_rotation(image_bgr, craft_result):
    """
    หมุนภาพให้กรอบ CRAFT ตรงแนวนอน (ใช้กรอบจาก final_polys/text_polys + ขอบยาวสุด).

    Args:
        image_bgr: ภาพ BGR (numpy)
        craft_result: dict จาก craft_detector.detect_text_and_rotate()

    Returns:
        (rotated_image, craft_angle_deg, rotate_deg) ถ้าสำเร็จ
        (None, None, None) ถ้าไม่มีกรอบหรือผิดพลาด
    """
    if image_bgr is None:
        return None, None, None
    box = get_craft_box_from_result(craft_result)
    if box is None:
        return None, None, None

    angle_edge, p1, p2 = compute_longest_edge_angle_and_points(box)

    # เมื่อมุมขอบยาวสุดใกล้ ~80° กรอบ CRAFT มักคลาดเคลื่อน — ลด |มุม| ลง 5° ก่อนหมุนจริง
    _ae_abs = abs(angle_edge)
    if 72.0 <= _ae_abs <= 88.0:
        _sign = 1.0 if angle_edge >= 0.0 else -1.0
        angle_edge = angle_edge - _sign * 5.0

    a = angle_edge
    if a > 90.0:
        rotate_deg = 180.0 - a
    elif a < -90.0:
        rotate_deg = -180.0 - a
    else:
        rotate_deg = -a

    try:
        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        rotated = cv2.warpAffine(image_bgr, M, (w, h))
        return rotated, angle_edge, rotate_deg
    except Exception:
        return None, angle_edge, rotate_deg
