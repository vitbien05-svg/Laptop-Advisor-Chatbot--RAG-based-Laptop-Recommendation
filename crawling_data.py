from pymongo import MongoClient
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import random
import os


client = MongoClient("mongodb://localhost:27017/")
db = client["LaptopDataDB"]
collection = db["laptops"]
brand_urls = [
    # "https://www.thegioididong.com/laptop-asus",
    "https://www.thegioididong.com/laptop-dell",
    "https://www.thegioididong.com/laptop-hp-compaq",
    "https://www.thegioididong.com/laptop-lenovo",
    "https://www.thegioididong.com/laptop-acer",
    "https://www.thegioididong.com/laptop-msi",
    "https://www.thegioididong.com/laptop-gigabyte",
]
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    url_page = "https://www.thegioididong.com"
    for craw_page in brand_urls:
        print(f"\n🚀 Đang cào hãng: {craw_page}")
        page.goto(craw_page)
        current_brand = craw_page.split("laptop-")[-1].upper()
        brand_products = []
        while True:
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.uniform(1000, 2000))
                btn = page.locator(".see-more-btn")
                if btn.count() == 0:
                    break
                btn.click()
                page.wait_for_timeout(random.uniform(1200, 2200))
            except:
                break
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select("ul.listproduct li[data-id]")
        print(len(items))
        count = 0
        # setup for crawling detail page, mapping key-value
        key_mapping = {
            "Card màn hình": "gpu",
            "RAM": "ram_info",
            "Loại RAM": "ram_type",
            "Hỗ trợ RAM tối đa": "max_ram_upgrade",
            "Ổ cứng": "storage_capacity",
            "Kích thước màn hình": "display_size",
            "Độ phân giải": "resolution",
            "Tấm nền": "panel_type",
            "Độ phủ màu": "color_gamut",
            "Tần số quét": "refresh_rate",
            "Công nghệ màn hình": "display_technology",
            "Cổng giao tiếp": "port",
            "Webcam": "webcam",
            "Thông tin Pin": "battery_specifications",
            "Kích thước": "physical_dimensions",
            "Chất liệu": "material",
            "Thời điểm ra mắt": "release_year",
        }
        for item in items:
            try:
                # id product
                item_id = item.get("data-id") if item.get("data-id") else None
                # link_product
                if item.select_one("a.main-contain"):
                    link_product = url_page + item.select_one("a.main-contain")["href"]
                else:
                    count += 1
                    link_product = None
                # name:product
                name_tag = item.select_one("p.product-title")
                name_product = name_tag.text.strip() if name_tag else None

                # price:
                price_tag = item.select_one("strong.price")
                price = price_tag.text.strip() if price_tag else None
                # url_img:
                img_tag = item.select_one("div.item-img.item-img_44 img")
                if img_tag:
                    img_product = img_tag.get("data-src") or img_tag.get("src")
                else:
                    img_product = None
                product = {
                    "brand": current_brand,
                    "id": item_id,
                    "url_product": link_product,
                    "name_product": name_product,
                    "price": price,
                    "img_product": img_product,
                    # cpu
                    "cpu_name": None,
                    "gpu": None,
                    # ram and storage
                    "ram_info": None,
                    "ram_type": None,
                    "max_ram_upgrade": None,
                    "storage_capacity": None,
                    # screen
                    "display_size": None,
                    "panel_type": None,
                    "resolution": None,
                    "color_gamut": None,
                    "refresh_rate": None,
                    "display_technology": None,
                    # port and some out :))
                    "port": None,
                    "webcam": None,
                    "battery_specifications": None,
                    "physical_dimensions": None,
                    "material": None,
                    "release_year": None,
                }
                # detail_crawling page:

                if product["url_product"]:
                    page.goto(product["url_product"])
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    waiting_time = random.uniform(1500, 4000)
                    page.wait_for_timeout(waiting_time)

                    detail_html = page.content()
                    soup = BeautifulSoup(detail_html, "html.parser")
                    content_page = soup.select_one("div.specification-item")
                    # print(content_page)
                    if content_page:
                        spec_items = content_page.select(".text-specifi li")
                        for item in spec_items:
                            asides = item.find_all("aside")
                            # first aside for key, second aside is value
                            if len(asides) == 2:
                                label_tag = asides[0].find("strong")
                                if label_tag:
                                    label = label_tag.text.strip().replace(":", "")
                                    value = asides[1].get_text(
                                        separator=", ", strip=True
                                    )
                                    # mapping
                                    if label in key_mapping:
                                        dict_key = key_mapping[label]
                                        product[dict_key] = value
                brand_products.append(product)
            except Exception as e:
                print("Error:", e)
        if brand_products:
            try:
                collection.insert_many(brand_products)
                print(f"[MongoDB] successfully save {len(brand_products)} samples")
            except Exception as eMongo:
                print(f"[MongoDB Error] can't save : {eMongo}")
        else:
            print("[MongoDB] Something wrong, this brand aren't find any products")
    browser.close()
    print("\n finished")
