"""
test_hard_cases.py — Test HÀNH VI RAG (luồng mới: luôn retrieve → CRAG → lọc → tư vấn).

Chạy run_agent THẬT trên các ca khó, kiểm:
  - CRAG có hỏi lại khi mơ hồ không (ca 1).
  - Ràng buộc bóc ra đúng không + có LOẠI hãng (né Acer) không (ca 2).
  - Parse '25 củ'/'RAM to' đúng không (ca 3).

Chạy:  python eval/test_hard_cases.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.chdir(_ROOT)
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

import chat_bot_cloud as cb

CASES = [
    ("Tư vấn mình mua laptop đi thực tập đồ với.",
     "Mơ hồ → CRAG phải HỎI LẠI, không trả danh sách máy."),
    ("Tư vấn máy nào màn đẹp tối cày Attack on Titan, ban ngày mang gọn nhẹ bỏ balo. "
     "Giá dưới 20, né hãng Acer ra nha.",
     "Phải LOẠI Acer + giá < 20; hiểu 'cày AoT'=giải trí, 'bỏ balo'=mỏng nhẹ."),
    ("Mình học AI, hay cắm Docker với lâu lâu chạy vài model nhẹ. Cần máy RAM to tí, tầm 25 củ đổ lại.",
     "Parse price_max=25, ram_min=16; hiểu 'học AI/Docker'=lập trình."),
]

for i, (q, expect) in enumerate(CASES, 1):
    print("=" * 80)
    print(f"CA {i}: {q}")
    print(f"KỲ VỌNG: {expect}")
    print("-" * 80)
    cons = cb.extract_constraints(cb.model, q)
    print(f"RÀNG BUỘC BÓC ĐƯỢC: price_max={cons.price_max}, price_min={cons.price_min}, "
          f"ram_min={cons.ram_min}, brand_include={cons.brand_include}, brand_exclude={cons.brand_exclude}")
    answer, products = cb.run_agent(q, [])
    is_clarify = not products
    print(f"CRAG hỏi lại (không có sản phẩm): {'CÓ' if is_clarify else 'KHÔNG'}")
    acer = [p["name"] for p in products if "acer" in (p.get("name", "") or "").lower()]
    if acer:
        print(f"   ⚠ Acer LỌT VÀO: {acer}")
    if products:
        print("SẢN PHẨM GỢI Ý:")
        for p in products:
            print(f"   • {p['name']} | {p.get('price_m','?')} triệu | RAM {p.get('ram')}GB | {p.get('gpu')}")
    print(f"\nTRẢ LỜI:\n{answer}\n")
