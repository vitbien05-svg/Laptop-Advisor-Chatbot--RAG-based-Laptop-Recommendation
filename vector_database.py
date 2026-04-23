from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
pd.set_option("display.max_colwidth", None)
import os


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
    # merge everything to become Full text
    page_content = " ".join(parts)

    metadata = {
        "id": str(row.get("id")),
        "brand": str(brand).lower(),
        "price_num": float(price / 1_000_000) if pd.notnull(price) else 0,
        "ram_gb": int(row.get("ram_info", 0)) if pd.notnull(row.get("ram_info")) else 0,
        "cpu": str(row.get("cpu_name", "")),
        "url": str(row.get("url_product", "")),
        "img": str(row.get("img_product", "")),
        "gpu_name": str(row.get("gpu_name", "")),
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
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


vectorstore = Chroma.from_documents(
    documents=documents,        
    embedding=embeddings,
    persist_directory="./vectorstore",
    ids=[doc.metadata["id"] for doc in documents],
)
print("Finish Vector Database!")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
