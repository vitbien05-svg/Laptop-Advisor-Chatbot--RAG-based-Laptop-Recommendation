from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


product = {
    # cpu
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
link_product = "https://www.thegioididong.com/laptop/macbook-air-13-inch-m5-16gb-512gb"
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(link_product)
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
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
                    value = asides[1].get_text(separator=", ", strip=True)
                    # mapping
                    if label in key_mapping:
                        dict_key = key_mapping[label]
                        product[dict_key] = value
    print(product)
