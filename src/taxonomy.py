"""
taxonomy.py — Gán nhãn use-case cho laptop bằng RULE trên specs đã sạch (Tầng A của enrichment).

Nguyên tắc thiết kế (giải thích được khi phỏng vấn):
- Ground truth = specs thô (facts) + taxonomy/rule ĐƯỢC ĐỊNH NGHĨA & KIỂM CHỨNG. Rule không
  phải chân lý tuyệt đối, mà là lớp gán nhãn có thể kiểm chứng; ngưỡng là lựa chọn thiết kế
  nên được document.
- Ngưỡng calibrate theo phân phối THỰC TẾ của catalog + quy ước 2 bài (macone.vn, tinhte.vn),
  KHÔNG theo "năm". Ngưỡng PHÂN LOẠI (máy này LÀ gì) ≠ ngưỡng KHUYẾN NGHỊ MUA MỚI.
- Multi-label: 1 máy có thể nhiều nhãn. null → KHÔNG gán. Đảm bảo mỗi máy ≥1 nhãn (default).

Chạy để xem phân phối nhãn (validate, chưa ghi DB):   python taxonomy.py
Ghi use_case_tags + price_tier vào MongoDB:            python taxonomy.py --write
"""
import argparse
import math
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ────────────────────────────── PARSERS ──────────────────────────────
def _num(x):
    """float an toàn; trả None nếu thiếu/NaN."""
    if x is None:
        return None
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def parse_storage_gb(text):
    """'512 GB SSD M.2', '1 TB SSD' → GB (int). None nếu không đọc được."""
    if not isinstance(text, str):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", text, re.I)
    if not m:
        return None
    val = float(m.group(1))
    return int(val * 1024) if m.group(2).upper() == "TB" else int(val)


def parse_display_inch(x):
    """15.6 (float) hoặc '15.6\"' → float."""
    if isinstance(x, (int, float)):
        return _num(x)
    if isinstance(x, str):
        m = re.search(r"(\d+(?:\.\d+)?)", x)
        return float(m.group(1)) if m else None
    return None


def color_srgb_equiv(text):
    """Quy các độ phủ màu về ~% sRGB tương đương để so ngưỡng.
    Ưu tiên sRGB rõ ràng; else DCI-P3/Adobe RGB (~tương đương); else NTSC*1.39 (72%NTSC≈100%sRGB).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    best = None

    def upd(v):
        nonlocal best
        if v is not None and (best is None or v > best):
            best = v

    for pat, factor in [
        (r"(\d+(?:\.\d+)?)\s*%?\s*sRGB", 1.0),
        (r"(\d+(?:\.\d+)?)\s*%?\s*DCI[- ]?P3", 1.0),
        (r"(\d+(?:\.\d+)?)\s*%?\s*Adobe", 1.0),
        (r"(\d+(?:\.\d+)?)\s*%?\s*NTSC", 1.39),
    ]:
        m = re.search(pat, text, re.I)
        if m:
            upd(float(m.group(1)) * factor)
    return best


def resolution_min_fhd(text):
    """True nếu độ phân giải ≥ Full HD (1920x1080)."""
    if not isinstance(text, str):
        return False
    if re.search(r"full\s*hd|1920", text, re.I):
        return True
    if re.search(r"2k|2560|2\.?8k|2880|3k|3200|4k|3840|3456|uhd|qhd", text, re.I):
        return True
    return False


def resolution_min_2k(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"2k|2560|2\.?8k|2880|3k|3200|4k|3840|3456|uhd|qhd|oled", text, re.I))


def parse_cpu(cpu_name):
    """→ dict(tier, brand, suffix, high_perf). suffix ∈ {U,P,HS,H,HX,None}."""
    out = {"tier": None, "brand": None, "suffix": None, "high_perf": False}
    if not isinstance(cpu_name, str) or not cpu_name.strip():
        return out
    s = cpu_name.strip()

    # Ryzen: R5 / R7 / Ryzen 5 ...
    m = re.search(r"\b(?:R|Ryzen)\s*(\d)\b", s, re.I)
    if m:
        out["brand"], out["tier"] = "ryzen", int(m.group(1))
    # Intel Core i-series
    m = re.search(r"\bi([3579])\b", s, re.I)
    if m:
        out["brand"], out["tier"] = "intel", int(m.group(1))
    # Intel Core (new) / Ultra: "Core 5 120U", "Ultra 9 185H"
    m = re.search(r"\b(?:Core|Ultra)\s*([3579])\b", s, re.I)
    if m:
        out["brand"] = "intel"
        out["tier"] = int(m.group(1))
        if re.search(r"Ultra", s, re.I):
            out["brand"] = "intel-ultra"

    # suffix: chữ cái đứng sau số model (13500H, 7520U, 7735HS, 14900HX)
    m = re.search(r"\d{3,5}\s*(HX|HS|H|U|P)\b", s, re.I)
    if m:
        out["suffix"] = m.group(1).upper()

    out["high_perf"] = out["suffix"] in {"H", "HX"} or (
        out["brand"] == "ryzen" and out["tier"] and out["tier"] >= 7
    ) or (out["tier"] == 9)
    return out


# ────────────────────────────── TAXONOMY RULES ──────────────────────────────
GAMING_GPU_RE = re.compile(r"\b(RTX|GTX|RX)\b", re.I)


def tag_laptop(doc):
    """Trả về list nhãn use-case (multi-label). null field → bỏ qua nhãn liên quan."""
    tags = []

    gpu_type = (doc.get("gpu_type") or "").lower()
    gpu_ram = _num(doc.get("gpu_ram"))
    gpu_name = doc.get("gpu_name") or ""
    ram = _num(doc.get("ram_info"))
    weight = _num(doc.get("weight_kg"))
    color = color_srgb_equiv(doc.get("color_gamut"))
    refresh = _num(doc.get("refresh_rate"))
    panel = (doc.get("panel_type") or "").upper()
    disp = parse_display_inch(doc.get("display_size"))
    storage = parse_storage_gb(doc.get("storage_capacity"))
    price = _num(doc.get("price"))  # VND
    cpu = parse_cpu(doc.get("cpu_name"))

    is_dedicated = "rời" in gpu_type
    is_integrated = "tích hợp" in gpu_type
    is_gaming_gpu = is_dedicated and bool(GAMING_GPU_RE.search(gpu_name)) and not re.search(r"\bMX\b", gpu_name, re.I)

    # 1. Văn phòng / học tập
    if is_integrated and ram is not None and ram >= 8:
        if price is None or price <= 20_000_000:
            tags.append("van_phong_hoc_tap")

    # 2. Mỏng nhẹ di động
    if weight is not None and weight <= 1.5:
        tags.append("mong_nhe_di_dong")

    # 3. Gaming (floor inclusive: mọi GPU game rời VRAM≥4, loại MX)
    if is_gaming_gpu and (gpu_ram is None or gpu_ram >= 4):
        tags.append("gaming")
        # 3b. Gaming cao cấp — VRAM là trục chính (RTX 4060+/8GB), KHÔNG lấy refresh
        if gpu_ram is not None and gpu_ram >= 8:
            tags.append("gaming_cao_cap")

    # 4. Đồ họa / sáng tạo (màn chuẩn màu là điểm phân biệt chính; overlap gaming OK)
    if color is not None and color >= 100 and ("IPS" in panel or "OLED" in panel) and ram is not None and ram >= 16:
        if is_dedicated or resolution_min_2k(doc.get("resolution")):
            tags.append("do_hoa_sang_tao")

    # 5. Lập trình / kỹ thuật
    if ram is not None and ram >= 16 and cpu["high_perf"] and (storage is None or storage >= 512):
        tags.append("lap_trinh_ky_thuat")

    # 6. Hiệu năng cao / workstation
    if (ram is not None and ram >= 32) or cpu["suffix"] == "HX" or cpu["tier"] == 9 or (gpu_ram is not None and gpu_ram >= 8):
        tags.append("hieu_nang_cao")

    # 7. Giải trí đa phương tiện — màn lớn + tấm nền tốt HƠN mức office cơ bản
    #    (siết: cần OLED / ≥2K / ≥120Hz / ≥16" để loại máy 15.6" FHD IPS 60Hz văn phòng)
    if (
        disp is not None and disp >= 15.6
        and ("IPS" in panel or "OLED" in panel)
        and resolution_min_fhd(doc.get("resolution"))
        and (
            "OLED" in panel
            or resolution_min_2k(doc.get("resolution"))
            or (refresh is not None and refresh >= 120)
            or disp >= 16
        )
    ):
        tags.append("giai_tri_da_phuong_tien")

    # Đảm bảo ≥1 nhãn
    if not tags:
        tags.append("van_phong_hoc_tap")

    return sorted(set(tags))


def price_tier(price_vnd):
    p = _num(price_vnd)
    if p is None:
        return None
    if p <= 15_000_000:
        return "pho_thong"
    if p <= 25_000_000:
        return "tam_trung"
    return "cao_cap"


# ────────────────────────────── MAIN ──────────────────────────────
def main():
    from pymongo import MongoClient
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Ghi use_case_tags + price_tier vào MongoDB")
    args = ap.parse_args()

    col = MongoClient("mongodb://localhost:27017/")["LaptopDataDB"]["laptop_cleaned"]
    docs = list(col.find({}))
    n = len(docs)

    tag_counter = Counter()
    label_per_machine = Counter()
    tier_counter = Counter()
    for d in docs:
        tags = tag_laptop(d)
        tag_counter.update(tags)
        label_per_machine[len(tags)] += 1
        tier_counter[price_tier(d.get("price"))] += 1
        if args.write:
            col.update_one(
                {"_id": d["_id"]},
                {"$set": {"use_case_tags": tags, "price_tier": price_tier(d.get("price"))}},
            )

    print(f"Tổng {n} máy.\n")
    print("== Phân phối NHÃN use-case (kiểm tra >60% hoặc <3% là kém discriminative) ==")
    for tag, c in tag_counter.most_common():
        print(f"  {tag:26s} {c:4d}  ({100*c/n:5.1f}%)")
    print("\n== Số nhãn / máy ==")
    for k in sorted(label_per_machine):
        print(f"  {k} nhãn: {label_per_machine[k]} máy")
    print("\n== Price tier ==")
    for t, c in tier_counter.most_common():
        print(f"  {t}: {c}")
    if args.write:
        print("\n✅ Đã ghi use_case_tags + price_tier vào laptop_cleaned.")
    else:
        print("\n(Chưa ghi DB — thêm --write để ghi.)")


if __name__ == "__main__":
    main()
