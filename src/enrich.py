"""
enrich.py — Tầng B: CHƯNG CẤT mô tả marketing → use-case summary khách quan (để embed).

- Input: description_raw (crawl) + specs + rule tags (taxonomy.py, Tầng A).
- Output: use_case_summary (2-4 câu, ngôn ngữ nhu cầu, BỎ cường điệu marketing),
  highlights (điểm mạnh thực), llm_tags (chọn từ taxonomy ĐÓNG để cross-check).
- Rào chắn: temperature=0, structured output, taxonomy đóng, chỉ dựa trên text+specs cung cấp
  (không bịa). Cross-check llm_tags với rule tags → log lệch. Rule tags vẫn là AUTHORITATIVE.
- Resume-able: bỏ qua doc đã có use_case_summary.

Test 10 máy:   python enrich.py --limit 10
Chạy full:     python enrich.py
"""
import argparse
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo import MongoClient

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

# Taxonomy ĐÓNG — LLM chỉ được chọn trong tập này (khớp taxonomy.py)
TAG_SET = [
    "van_phong_hoc_tap",
    "mong_nhe_di_dong",
    "gaming",
    "gaming_cao_cap",
    "do_hoa_sang_tao",
    "lap_trinh_ky_thuat",
    "hieu_nang_cao",
    "giai_tri_da_phuong_tien",
]


class Distilled(BaseModel):
    use_case_summary: str = Field(
        description="2-4 câu tiếng Việt KHÁCH QUAN mô tả máy phù hợp NHU CẦU nào và điểm mạnh "
        "thực tế. BỎ từ ngữ cường điệu marketing ('đáng kinh ngạc', 'chớp mắt'). Chỉ dựa vào "
        "mô tả + specs được cung cấp, KHÔNG bịa thông số."
    )
    highlights: list[str] = Field(description="3-5 điểm mạnh ngắn gọn, dựa trên specs thật.")
    llm_tags: list[str] = Field(description=f"Các nhãn use-case phù hợp, CHỈ chọn trong: {TAG_SET}")


SYSTEM = (
    "Bạn là biên tập viên chắt lọc thông tin laptop. Nhiệm vụ: đọc đoạn mô tả (văn quảng cáo) "
    "và bảng specs, rồi viết lại thành tóm tắt NHU CẦU khách quan, trung thực với specs. "
    "Tuyệt đối không thêm thông số không có trong dữ liệu. Chỉ chọn nhãn trong danh sách cho sẵn."
)


def build_user_prompt(doc):
    specs = {
        k: doc.get(k)
        for k in ["name_product", "price", "cpu_name", "ram_info", "gpu_type", "gpu_ram",
                  "gpu_name", "storage_capacity", "display_size", "panel_type", "resolution",
                  "color_gamut", "refresh_rate", "weight_kg"]
    }
    return (
        f"SPECS: {specs}\n"
        f"RULE_TAGS (tham chiếu): {doc.get('use_case_tags')}\n\n"
        f"MÔ TẢ:\n{doc.get('description_raw','')[:4000]}"
    )


def main():
    from langchain_openai import ChatOpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    col = MongoClient("mongodb://localhost:27017/")["LaptopDataDB"]["laptop_cleaned"]
    q = {"description_raw": {"$nin": [None, ""]}}
    if not args.overwrite:
        q["use_case_summary"] = {"$exists": False}
    docs = list(col.find(q))
    if args.limit:
        docs = docs[: args.limit]
    print(f"Chưng cất {len(docs)} máy bằng {args.model}...")

    llm = ChatOpenAI(model=args.model, temperature=0).with_structured_output(Distilled)

    ok, mismatch, fail = 0, 0, 0
    for i, d in enumerate(docs, 1):
        try:
            res: Distilled = llm.invoke(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_user_prompt(d)}]
            )
            # cross-check: giữ nhãn LLM hợp lệ + không mâu thuẫn rule
            rule_tags = set(d.get("use_case_tags", []))
            llm_tags = {t for t in res.llm_tags if t in TAG_SET}
            disagree = sorted(llm_tags ^ rule_tags)
            if disagree:
                mismatch += 1

            col.update_one(
                {"_id": d["_id"]},
                {"$set": {
                    "use_case_summary": res.use_case_summary.strip(),
                    "highlights": res.highlights,
                    "llm_tags": sorted(llm_tags),
                    "tag_disagreement": disagree,  # để review; rule tags vẫn authoritative
                }},
            )
            ok += 1
            if i % 20 == 0 or args.limit:
                print(f"[{i}/{len(docs)}] OK  {d.get('name_product','')[:40]}")
                if args.limit:
                    print(f"    summary: {res.use_case_summary[:160]}")
                    if disagree:
                        print(f"    ⚠ lệch rule↔llm: {disagree}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(docs)}] FAIL {d.get('name_product','')[:40]} :: {type(e).__name__}: {str(e)[:90]}")

    print(f"\nXong. OK={ok} (lệch rule↔llm: {mismatch}) FAIL={fail} / {len(docs)}")


if __name__ == "__main__":
    main()
