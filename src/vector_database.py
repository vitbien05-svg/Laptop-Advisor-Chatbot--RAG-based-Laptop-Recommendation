from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
pd.set_option("display.max_colwidth", None)
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Ánh xạ nhãn use-case → cụm từ "ngôn ngữ nhu cầu" để nhúng vào page_content (giúp semantic)
TAG_PHRASES = {
    "van_phong_hoc_tap": "phù hợp văn phòng, học tập, sinh viên",
    "mong_nhe_di_dong": "mỏng nhẹ, dễ mang đi, tính di động cao",
    "gaming": "chơi game, gaming",
    "gaming_cao_cap": "gaming cao cấp, chơi game nặng đồ họa cao",
    "do_hoa_sang_tao": "thiết kế đồ họa, sáng tạo nội dung, dựng phim, chỉnh ảnh",
    "lap_trinh_ky_thuat": "lập trình, code, kỹ thuật, IT",
    "hieu_nang_cao": "hiệu năng cao, tác vụ nặng, workstation",
    "giai_tri_da_phuong_tien": "giải trí, xem phim, đa phương tiện",
}


def create_laptop_document(row):
    parts = []

    name = row.get("name_product")
    brand = row.get("brand")
    if pd.notnull(name) and pd.notnull(brand):
        parts.append(f"Laptop {name} thương hiệu {brand}.")

    price = row.get("price")
    if pd.notnull(price) and price > 0:
        parts.append(f"Giá: {price/1_000_000:.2f} triệu Đồng.")

    # make cpu, ram, rom, gpu into group to easily manage
    perf = []
    if pd.notnull(row.get("cpu_name")):
        perf.append(f"Chip xử lí {row.get('cpu_name')}")
    if pd.notnull(row.get("ram_info")):
        perf.append(f"Bộ nhớ RAM {row.get('ram_info')}GB {row.get('ram_type', '')}")
    if pd.notnull(row.get("max_ram_upgrade")):
        perf.append(f"Khả năng nâng cấp ram tối đa là {row.get('max_ram_upgrade')} GB")
    if pd.notnull(row.get("gpu_type")):
        perf.append(f"card đồ họa của máy là card {row.get('gpu_type')}")
    if pd.notnull(row.get("gpu_ram")):
        perf.append(f" card có bộ nhớ là {row.get('gpu_ram')}GB")
    if pd.notnull(row.get("gpu_name")):
        perf.append(f"{row.get('gpu_name')}")
    if pd.notnull(row.get("storage_capacity")):
        perf.append(f"ổ cứng trong có dung lượng là {row.get('storage_capacity')}")

    if perf:
        parts.append("Cấu hình laptop gồm: " + ", ".join(perf) + ".")

    # 3. make all infomation about display into group to easily manage
    disp = []
    if pd.notnull(row.get("display_size")):
        disp.append(f"{row.get('display_size')} inch")
    if pd.notnull(row.get("resolution")):
        disp.append(row.get("resolution"))
    if pd.notnull(row.get("panel_type")):
        disp.append(row.get("panel_type"))
    if pd.notnull(row.get("refresh_rate")):
        disp.append(f"{int(row['refresh_rate'])}Hz")
    if pd.notnull(row.get("color_gamut")):
        disp.append(f"độ phủ màu {row.get('color_gamut')}")
    if pd.notnull(row.get("webcam")):
        disp.append(f"Máy được tích hợp Webcam là {row.get('webcam')}.")

    if disp:
        parts.append("Về Phần Màn hình laptop: " + ", ".join(disp) + ".")

    # 4. Nhóm Thiết kế & Năm ra mắt (Theo ý ông muốn thêm vào)
    design = []
    if pd.notnull(row.get("material")):
        design.append(f"vỏ máy tính làm bằng {row.get('material')}")
    if pd.notnull(row.get("weight_kg")):
        design.append(f"cân nặng {row.get('weight_kg')}kg")
    if pd.notnull(row.get("release_year")):
        design.append(f"Máy ra mắt năm {int(row['release_year'])}")

    if design:
        parts.append("Thiết kế: " + ", ".join(design) + ".")

    # 5. port
    if pd.notnull(row.get("port")):
        parts.append(f"Máy gồm các cổng kết nối sau: {row.get('port')}.")

    # 6. ENRICHMENT — nhãn use-case (ngôn ngữ nhu cầu) + summary chưng cất → cải thiện semantic
    tags = row.get("use_case_tags") or []
    if isinstance(tags, (list, tuple)) and len(tags) > 0:
        phrases = [TAG_PHRASES.get(t, t) for t in tags]
        parts.append("Nhu cầu phù hợp: " + "; ".join(phrases) + ".")
    summary = row.get("use_case_summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())

    # merge everything to become Full text
    page_content = " ".join(parts)

    metadata = {
        "id": str(row.get("id")),
        "name": str(name) if pd.notnull(name) else "",
        "brand": str(brand).lower(),
        "price_num": float(price / 1_000_000) if pd.notnull(price) else 0,
        "ram_gb": int(row.get("ram_info", 0)) if pd.notnull(row.get("ram_info")) else 0,
        "cpu": str(row.get("cpu_name", "")),
        "url": str(row.get("url_product", "")),
        "img": str(row.get("img_product", "")),
        "gpu_name": str(row.get("gpu_name", "")),
        "use_case_tags": ",".join(tags) if isinstance(tags, (list, tuple)) else "",
        "price_tier": str(row.get("price_tier", "")),
    }
    return Document(page_content=page_content, metadata=metadata)


client = MongoClient("mongodb://localhost:27017/")
db = client["LaptopDataDB"]
collection = db["laptop_cleaned"]
data = pd.DataFrame(list(collection.find({}, {"_id": 0})))
documents = []
for i, item in data.iterrows():
    try:
        doc = create_laptop_document(item)
        if doc and doc.page_content and doc.page_content.strip():
            documents.append(doc)
        else:
            print(f"Row {i} bị rỗng")
    except Exception as e:
        print(f"error row {i}: {e}")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = OpenAIEmbeddings(model ="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=documents,        
    embedding=embeddings,
    persist_directory="./vectorstore",
    ids=[doc.metadata["id"] for doc in documents],

)
print("Finish Vector Database!")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
