"""
crawl_description.py — Bổ sung TEXT mô tả ("Thông tin sản phẩm") cho từng laptop.

Data specs đã có sẵn trong MongoDB (laptop_cleaned). Script CHỈ crawl thêm khối mô tả tự nhiên
ở trang chi tiết TGDD (selector: div.text-detail) → field `description_raw`.
Async, chạy SONG SONG nhiều trang để nhanh. Resume-able: bỏ qua doc đã có description_raw.

Test:            python crawl_description.py --limit 5
Full (nhanh):    python crawl_description.py --concurrency 5
"""
import argparse
import asyncio
import random
import sys

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from pymongo import MongoClient

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DESC_SELECTOR = "div.text-detail"
COLLECTION = "laptop_cleaned"
BLOCK = "**/*.{png,jpg,jpeg,webp,gif,svg,css,woff,woff2,ttf,mp4,avif,ico}"


def extract_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    block = soup.select_one(DESC_SELECTOR)
    if not block:
        return ""
    parts = [el.get_text(" ", strip=True) for el in block.select("p, h2, h3, li")]
    return "\n".join(p for p in parts if p).strip()


async def _block_route(route):
    await route.abort()


async def worker(wid, queue, col, browser, progress):
    page = await browser.new_page()
    await page.route(BLOCK, _block_route)
    while True:
        doc = await queue.get()
        if doc is None:
            queue.task_done()
            break
        try:
            await page.goto(doc["url_product"], timeout=30000, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector(DESC_SELECTOR, timeout=6000)
            except Exception:
                pass
            desc = extract_description(await page.content())
            col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"description_raw": desc, "description_len": len(desc)}},
            )
            progress["done"] += 1
            if not desc:
                progress["empty"] += 1
            if progress["done"] % 20 == 0:
                print(f"  ...{progress['done']}/{progress['total']} (empty={progress['empty']} fail={progress['fail']})")
        except Exception as e:
            progress["fail"] += 1
            print(f"  FAIL {doc.get('name_product','')[:40]} :: {type(e).__name__}: {str(e)[:70]}")
        await asyncio.sleep(random.uniform(0.2, 0.6))
        queue.task_done()
    await page.close()


async def main_async(args):
    client = MongoClient("mongodb://localhost:27017/")
    col = client["LaptopDataDB"][COLLECTION]
    q = {"url_product": {"$nin": [None, ""]}}
    if not args.overwrite:
        q["description_raw"] = {"$exists": False}
    docs = list(col.find(q, {"_id": 1, "url_product": 1, "name_product": 1}))
    if args.limit:
        docs = docs[: args.limit]
    print(f"Cần crawl {len(docs)} doc, song song {args.concurrency} trang.")
    if not docs:
        return

    progress = {"done": 0, "empty": 0, "fail": 0, "total": len(docs)}
    queue = asyncio.Queue()
    for d in docs:
        queue.put_nowait(d)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        workers = [
            asyncio.create_task(worker(i, queue, col, browser, progress))
            for i in range(args.concurrency)
        ]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
        await browser.close()

    print(f"\nXong. OK={progress['done']-progress['empty']} EMPTY={progress['empty']} "
          f"FAIL={progress['fail']} / {len(docs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
