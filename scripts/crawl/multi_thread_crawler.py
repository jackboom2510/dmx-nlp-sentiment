#!/usr/bin/env python3
"""
Crawl review Điện Máy Xanh -> data/raw/reviews.jsonl  (BẢN ĐA LUỒNG, ít worker)

Dựa trên crawl_dmx_reviews.py (1 luồng): toàn bộ logic crawl giữ nguyên, chỉ thay
main() bằng nhiều worker + thêm khóa ghi file, cờ dừng chung khi bị chặn.

Cài đặt:
    pip install requests lxml playwright
    playwright install chromium

Cách chạy:
    python crawl_dmx_reviews_mt.py --test 10 --workers 3 --debug     # thử 10 sản phẩm (ghi ra *_test_mt)
    python crawl_dmx_reviews_mt.py --workers 2                       # chạy toàn bộ, tự resume khi bị ngắt
    python crawl_dmx_reviews_mt.py --inspect <link>/danh-gia         # xem HTML để tìm selector

Quy tắc: mỗi worker nghỉ 2-4s giữa các lần tải/bấm, không gọi trực tiếp /aj/,
dùng IP thật. Bất kỳ worker nào gặp 403/429/captcha -> TẤT CẢ dừng và báo
(state đã được lưu, chạy lại để tiếp tục). Số worker tối đa: MAX_WORKERS.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import logging
import queue
import threading
from types import SimpleNamespace
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from lxml import etree
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ----------------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------------
BASE = "https://www.dienmayxanh.com"
SITEMAP_URL = f"{BASE}/newsitemap/sitemap-product"
ROBOTS_URL = f"{BASE}/robots.txt"

# Ngành hàng bị robots.txt cấm -> bỏ
DISALLOWED = [
    "/may-tinh-tin-hoc",
    "/balo-laptop",
    "/loa-nghe-nhac-da-nang",
    "/may-nghe-nhac-mp3",
    "/tu-dien-dien-tu",
    "/may-mp4-mp5",
]

# Ngành hàng cần lấy (slug đầu tiên của URL). Sửa/thêm cho khớp thực tế sitemap.
TARGET_CATEGORIES = {
    "may-giat": "Máy giặt",
    "tu-lanh": "Tủ lạnh",
    "may-lanh": "Máy lạnh",
    "tivi": "Tivi",
    "noi-com-dien": "Nồi cơm điện",
    "may-loc-nuoc": "Máy lọc nước",
}

MAX_PER_PRODUCT = 300  # tối đa review / sản phẩm
PER_STAR_CAP = 100  # tối đa mỗi mức 1★,2★,3★ khi lọc theo sao
MAX_CLICKS_PER_PASS = 40  # chống lặp vô hạn
TARGET_TOTAL = 10000  # dừng khi đủ
DELAY = (2.0, 4.0)  # nghỉ giữa các lần tải trang / bấm nút (MỖI worker)
MAX_WORKERS = 8  # trần an toàn; chỉ nâng khi nhóm đã thống nhất đổi quy tắc 1 luồng

OUT_DIR = "data/raw"
LOG_FILE = "data/logs/crawl.log"
SITEMAP_REPORT = f"{OUT_DIR}/sitemap_report.json"
DEBUG_MODE = False
OUT_FILE = f"{OUT_DIR}/reviews.jsonl"
STATE_FILE = f"{OUT_DIR}/crawled_products.json"
URLS_CACHE = f"{OUT_DIR}/product_urls.json"
DEBUG_DIR = "data/debug"
log = logging.getLogger("dmx")

# !!! CÁC SELECTOR DƯỚI ĐÂY LÀ ƯỚC ĐOÁN - PHẢI KIỂM TRA BẰNG --inspect (bước 4) !!!
# Mỗi mục là danh sách ứng viên, script thử lần lượt và dùng cái đầu tiên khớp.
SEL = {
    # mỗi khối 1 review
    "item": [
        "li.comment__item",
        "div.comment__item",
        "li.par",
        "div.cmt-item",
        "ul.comment-list > li",
    ],
    # nội dung review (bên trong item)
    "content": [".cmt-txt", ".comment__content", ".cmt-content", "p.cmt-txt"],
    # phần trả lời của nhân viên -> sẽ bị loại khỏi nội dung
    "staff_reply": [
        ".reply",
        ".replycomment",
        ".cmt-reply",
        ".listreply",
        ".cmt-command",
        ".admin",
        ".qtv",
    ],
    # ngày
    "date": [".cmt-time", ".comment__time", "time", ".cmtd", ".cmt-date"],
    # sao "đầy" (đếm số phần tử) - icon sao vàng
    "star_filled": [
        ".iconcmt-starbuy",
        ".icon-star.active",
        ".star-fill",
        "i.iconcmt-starbuy",
    ],
    # nút "Xem thêm" nạp thêm review
    "load_more": [
        "a.btn-viewmore",
        "a.readmore",
        "button:has-text('Xem thêm')",
        "a:has-text('Xem thêm đánh giá')",
        "a:has-text('Xem thêm')",
    ],
    # nút trang tiếp theo (nếu là phân trang)
    # "next_page": ["a.next", "li.next a", "a[aria-label='Next']", "a:has-text('Trang sau')"],
    "next_page": [".pagcomment span.active + a"],
    # bộ lọc theo sao: {n} = 1..5
    "star_filter": [
        ".filter-left .filter-list li:nth-child({k})",
        ".filter-left .filter-list li:text-is('{n}')",
    ],
    "filter_all": [".filter-left .filter-list li:nth-child(1)"],
    # điểm trung bình / tổng số đánh giá (chỉ để log, không bắt buộc)
    "avg_rating": [".point-average", ".rating-average", ".average-rating"],
}

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1m\033[31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        # Tạo format động có chèn mã màu xung quanh [%(levelname)s]
        fmt = f"%(asctime)s [%(threadName)s] [{log_color}%(levelname)s{self.RESET}] %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logging(debug=False):
    global DEBUG_MODE
    DEBUG_MODE = debug

    # 1. Tạo thư mục chứa file log nếu chưa có
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # 2. XÓA HANDLER CŨ (Tránh lỗi trùng lặp/in đúp log)
    if log.hasHandlers():
        log.handlers.clear()

    # Set level thấp nhất cho Logger gốc để nó nhận mọi loại log
    log.setLevel(logging.DEBUG)

    # 3. Cấu hình File Handler (Ghi log sạch ra file, luôn ghi từ DEBUG trở lên)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s", "%H:%M:%S"
    )
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # Luôn ghi nhận DEBUG vào file
    fh.setFormatter(file_fmt)

    # 4. Cấu hình Stream Handler (In log có màu ra Console dựa theo chế độ debug)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(ColoredFormatter())  # Áp dụng formatter có màu ở đây

    # 5. Thêm các handler vào logger
    log.addHandler(fh)
    log.addHandler(sh)


def totals(stats):
    total = sum(stats.values())
    low = sum(v for (c, s), v in stats.items() if s in (1, 2))
    three = sum(v for (c, s), v in stats.items() if s == 3)
    return total, low, three


# ----------------------------------------------------------------------------
# TIỆN ÍCH
# ----------------------------------------------------------------------------
class BlockedError(Exception):
    """403/429/captcha -> dừng toàn bộ."""


def sleep():
    time.sleep(random.uniform(*DELAY))


PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?84|0)[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\d)"
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def scrub(text):
    """Xoá số điện thoại/email nếu người dùng lỡ viết vào nội dung."""
    text = PHONE_RE.sub("[SĐT]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_date(raw):
    """Chuyển '12/03/2025' hoặc '3 tháng trước' thành ISO (yyyy-mm-dd). Không được -> None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            return None
    m = re.search(r"(\d+)\s*(phút|giờ|ngày|tuần|tháng|năm)\s*trước", raw)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"phút": 0, "giờ": 0, "ngày": 1, "tuần": 7, "tháng": 30, "năm": 365}[
            unit
        ]
        return (datetime.now() - timedelta(days=n * days)).date().isoformat()
    return None


def review_id(product_code, rating, date_raw, content):
    key = f"{product_code}|{rating}|{date_raw}|{content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# BƯỚC 1-2: LẤY DANH SÁCH SẢN PHẨM TỪ SITEMAP
# ----------------------------------------------------------------------------
def load_robots():
    rp = RobotFileParser()
    try:
        r = requests.get(ROBOTS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        rp.parse(r.text.splitlines())
    except Exception as e:
        log.error(
            f"! Không đọc được robots.txt ({e}); chỉ dùng danh sách cấm thủ công."
        )
        rp = None
    return rp


def is_allowed(url, rp):
    path = urlparse(url).path
    if any(path.startswith(p) for p in DISALLOWED):
        return False
    if rp is not None and not rp.can_fetch("*", url):
        return False
    return True


def fetch_locs(url):
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code in (403, 429):
        raise BlockedError(f"Sitemap trả về {r.status_code}: {url}")
    r.raise_for_status()
    root = etree.fromstring(r.content)
    return [
        e.text.strip()
        for e in root.iter()
        if isinstance(e.tag, str) and e.tag.endswith("loc") and e.text
    ]


def fetch_sitemap(url, report, depth=0):
    t0 = time.time()
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    log.info(
        f"[sitemap] HTTP {r.status_code} | {len(r.content)/1024:.0f} KB | {time.time()-t0:.1f}s | {url}"
    )
    if r.status_code in (403, 429):
        report[url] = {"status": f"blocked_{r.status_code}"}
        raise BlockedError(f"Sitemap trả về {r.status_code}: {url}")
    r.raise_for_status()

    root = etree.fromstring(r.content)
    tag = etree.QName(root).localname
    locs = [
        e.text.strip()
        for e in root.iter()
        if isinstance(e.tag, str) and etree.QName(e).localname == "loc" and e.text
    ]

    if tag == "sitemapindex":
        log.info(f"[sitemap] Index có {len(locs)} sitemap con")
        report[url] = {"status": "index", "children": len(locs)}
        out = []
        if depth >= 2:
            return out
        for i, child in enumerate(locs, 1):
            log.info(f"[sitemap] ({i}/{len(locs)}) đang quét {child}")
            time.sleep(random.uniform(*DELAY))
            try:
                out.extend(fetch_sitemap(child, report, depth + 1))
            except BlockedError:
                raise
            except Exception as e:
                log.error(f"[sitemap] LỖI {child}: {e}")
                report[child] = {"status": "error", "error": str(e)}
        return out

    # urlset
    cats = Counter()
    for u in locs:
        parts = [p for p in urlparse(u).path.split("/") if p]
        if parts:
            cats[parts[0]] += 1
    target = sum(v for c, v in cats.items() if c in TARGET_CATEGORIES)
    report[url] = {
        "status": "ok",
        "urls": len(locs),
        "target_urls": target,
        "top_categories": cats.most_common(5),
    }
    log.info(f"[sitemap]   -> OK: {len(locs)} URL, {target} thuộc ngành mục tiêu")
    return locs


def get_product_urls(refresh=False):
    if os.path.exists(URLS_CACHE) and not refresh:
        with open(URLS_CACHE, encoding="utf-8") as f:
            data = [tuple(x) for x in json.load(f)]
        log.info(f"[sitemap] Dùng cache {URLS_CACHE}: {len(data)} sản phẩm")
        return data

    rp = load_robots()
    report = {}
    try:
        raw = fetch_sitemap(SITEMAP_URL, report)
    finally:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(SITEMAP_REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)

    products, seen = [], set()
    for u in raw:
        u = u.split("?")[0].rstrip("/")
        parts = [p for p in urlparse(u).path.split("/") if p]
        if len(parts) < 2:
            continue
        cat = parts[0]
        if cat not in TARGET_CATEGORIES or not is_allowed(u, rp) or u in seen:
            continue
        seen.add(u)
        products.append((u, cat))

    bad = [k for k, v in report.items() if v["status"] not in ("ok", "index")]
    ok = sum(1 for v in report.values() if v["status"] == "ok")
    log.info(
        f"[sitemap] Xong: {ok} sitemap OK, {len(bad)} lỗi | {len(products)} sản phẩm mục tiêu"
    )
    for c, n in Counter(c for _, c in products).most_common():
        log.info(f"[sitemap]   {TARGET_CATEGORIES[c]}: {n}")
    for c in TARGET_CATEGORIES:
        if not any(pc == c for _, pc in products):
            log.warning(
                f"[sitemap] Ngành '{c}' có 0 sản phẩm -> kiểm tra lại slug trong TARGET_CATEGORIES"
            )

    if bad:
        log.warning(
            f"[sitemap] {len(bad)} sitemap lỗi -> KHÔNG ghi cache. Xem {SITEMAP_REPORT}, chạy lại với --refresh-sitemap"
        )
    else:
        with open(URLS_CACHE, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False)
    return products


def prioritize(products, seed=42):
    """Xáo trộn trong từng ngành rồi xen kẽ giữa các ngành để cân bằng dữ liệu."""
    rnd = random.Random(seed)
    by_cat = defaultdict(list)
    for u, c in products:
        by_cat[c].append((u, c))
    for lst in by_cat.values():
        rnd.shuffle(lst)
    out, i = [], 0
    while any(by_cat.values()):
        for c in list(by_cat):
            if by_cat[c]:
                out.append(by_cat[c].pop())
        i += 1
    return out


# ----------------------------------------------------------------------------
# PLAYWRIGHT: THAO TÁC TRÊN TRANG
# ----------------------------------------------------------------------------
class Crawler:
    def __init__(self, page, stop=None):
        self.page = page
        self.stop = stop  # threading.Event dùng chung giữa các worker (None = 1 luồng)
        self.blocked_status = None
        page.on("response", self._on_response)

    # ------------------------------------------------------------------
    # Chặn / captcha
    # ------------------------------------------------------------------
    def _on_response(self, resp):
        try:
            if (
                resp.status in (403, 429)
                and resp.url.startswith(BASE)
                and resp.request.resource_type in ("document", "xhr", "fetch")
            ):
                self.blocked_status = resp.status
        except Exception:
            pass

    def check_blocked(self):
        if self.stop is not None and self.stop.is_set():
            raise BlockedError("Dừng: worker khác đã gặp chặn")
        if self.blocked_status:
            raise BlockedError(f"Nhận HTTP {self.blocked_status}")

        title = (self.page.title() or "").lower()
        if "just a moment" in title or "access denied" in title:
            raise BlockedError("Trang thử thách / chặn truy cập")

        # Captcha chỉ tính là chặn khi đang hiển thị VÀ trang không có review
        challenge = self.page.locator(
            "iframe[src*='recaptcha']:visible, iframe[src*='hcaptcha']:visible, "
            "iframe[src*='challenge']:visible, #cf-challenge-running, .g-recaptcha:visible"
        )
        if challenge.count() > 0 and self.item_count() == 0:
            raise BlockedError("Phát hiện captcha đang hiển thị")

    # ------------------------------------------------------------------
    # Điều hướng
    # ------------------------------------------------------------------
    def goto(self, url):
        """Mở URL, trả về mã HTTP. 403/429 -> BlockedError; 404 không phải lỗi chặn."""
        resp = self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
        status = resp.status if resp is not None else None
        if status in (403, 429):
            raise BlockedError(f"Nhận HTTP {status} khi mở {url}")
        try:
            self.page.wait_for_load_state("networkidle", timeout=15000)
        except PWTimeout:
            pass
        sleep()
        self.check_blocked()
        return status

    def open_reviews(self, url):
        """
        Ưu tiên <url>/danh-gia; không dùng được thì quay về trang gốc.
        Trả về 'danh-gia' | 'product_page' | None.
        """
        product = url.split("?")[0].rstrip("/")
        status = self.goto(f"{product}/danh-gia")
        final = self.page.url.split("?")[0].rstrip("/")

        if (
            status
            and status < 400
            and final.endswith("/danh-gia")
            and self.wait_items(6000)
        ):
            return "danh-gia"

        log.info(
            f"   /danh-gia không dùng được (HTTP {status}, url cuối: {final}) -> thử trang gốc"
        )

        # Nếu /danh-gia đã redirect về đúng trang gốc thì không cần tải lại
        if final != product or (status and status >= 400):
            self.goto(product)

        self.scroll_to_reviews()
        if self.wait_items(8000):
            return "product_page"
        return None

    def scroll_to_reviews(self):
        """Cuộn tới vùng đánh giá để kích hoạt lazy-load."""
        sec = self.first_selector("load_more") or self.first_selector("review_section")
        try:
            if not sec:
                raise RuntimeError("không có mốc để cuộn")
            self.page.locator(sec).first.scroll_into_view_if_needed(timeout=5000)
        except Exception:
            for _ in range(8):
                self.page.mouse.wheel(0, 700)
                time.sleep(0.4)
        sleep()
        self.check_blocked()

    # ------------------------------------------------------------------
    # Đếm / chờ phần tử
    # ------------------------------------------------------------------
    def first_selector(self, key, **fmt):
        """Selector ứng viên đầu tiên có phần tử trên trang (kể cả phần tử đang ẩn)."""
        for s in SEL.get(key, []):
            s = s.format(**fmt) if fmt else s
            try:
                if self.page.locator(s).count() > 0:
                    return s
            except Exception:
                continue
        return None

    def item_count(self):
        s = self.first_selector("item")
        return self.page.locator(s).count() if s else 0

    def wait_items(self, timeout_ms):
        combined = ", ".join(SEL["item"])
        try:
            # state="attached": chấp nhận cả review đang bị ẩn (display: none)
            self.page.wait_for_selector(combined, timeout=timeout_ms, state="attached")
            return True
        except PWTimeout:
            return False

    def total_hint(self):
        """Tổng số đánh giá đọc từ link 'Xem N đánh giá' (link có thể đang ẩn)."""
        sels = SEL.get("show_all_link", [])
        if not sels:
            return None
        loc = self.page.locator(sels[0])
        if loc.count():
            m = re.search(r"\d+", loc.first.text_content() or "")
            return int(m.group()) if m else None
        return None

    def click_first_visible(self, key, **fmt):
        for s in SEL.get(key, []):
            s = s.format(**fmt) if fmt else s
            try:
                loc = self.page.locator(s)
                if loc.count() and loc.first.is_visible():
                    loc.first.scroll_into_view_if_needed()
                    loc.first.click(timeout=8000)
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Bộ lọc sao (.filter-left)
    # ------------------------------------------------------------------
    def _wait_filter_applied(self, expected):
        """expected: '1'..'5' hoặc 'Tất cả'. Chờ li.current khớp."""
        try:
            self.page.wait_for_function(
                """(exp) => {
                    const c = document.querySelector('.filter-left .filter-list li.current');
                    if (!c) return false;
                    const t = c.innerText.trim();
                    return exp === 'Tất cả' ? t.startsWith('Tất cả') : t.startsWith(exp);
                }""",
                arg=expected,
                timeout=10000,
            )
            return True
        except PWTimeout:
            return False

    def apply_star_filter(self, n):
        # k = 7 - n: Tất cả=1, 5★=2, 4★=3, 3★=4, 2★=5, 1★=6
        if not self.click_first_visible("star_filter", n=n, k=7 - n):
            return False
        ok = self._wait_filter_applied(str(n))
        sleep()
        self.check_blocked()
        return ok

    def reset_filter(self):
        if not self.click_first_visible("filter_all"):
            return False
        ok = self._wait_filter_applied("Tất cả")
        sleep()
        self.check_blocked()
        return ok

    # ------------------------------------------------------------------
    # Đọc review
    # ------------------------------------------------------------------
    def extract(self):
        """
        Đọc tất cả review đang có trong DOM (kể cả review ẩn).
        Không đọc tên người dùng; loại phần trả lời của nhân viên khỏi nội dung.
        """
        args = {
            k: SEL.get(k, [])
            for k in ("item", "content", "staff_reply", "date", "star_filled")
        }
        js = """
        (sel) => {
          const pick = (el, list) => {
            for (const s of list) { if (el.querySelector(s)) return s; }
            return null;
          };
          let itemSel = null;
          for (const s of sel.item) { if (document.querySelector(s)) { itemSel = s; break; } }
          if (!itemSel) return [];
          return Array.from(document.querySelectorAll(itemSel)).map(it => {
            const clone = it.cloneNode(true);
            sel.staff_reply.forEach(s => clone.querySelectorAll(s).forEach(n => n.remove()));
            let content = '';
            const cs = pick(clone, sel.content);
            if (cs) content = clone.querySelector(cs).textContent || '';   // textContent: đọc được cả phần tử ẩn
            let date = '';
            const ds = pick(clone, sel.date);
            if (ds) date = clone.querySelector(ds).textContent || '';
            let stars = 0;
            for (const s of sel.star_filled) {
              const n = clone.querySelectorAll(s).length;
              if (n > 0) { stars = n; break; }
            }
            return { id: it.id || '', content: content, date: date, stars: stars };
          });
        }
        """
        return self.page.evaluate(js, args)

    # ------------------------------------------------------------------
    # Nạp thêm / chuyển trang, gọi save_fn sau mỗi lần đọc
    # ------------------------------------------------------------------
    def crawl_pages(self, cap, save_fn):
        """
        Đọc trang hiện tại, rồi bấm 'Xem thêm' hoặc trang kế tiếp, lặp đến khi
        đủ cap hoặc hết nút. save_fn(batch) trả về số review MỚI đã lưu.
        """
        items_sel = ", ".join(SEL["item"])
        total, stall = 0, 0

        for _ in range(MAX_CLICKS_PER_PASS):
            batch = self.extract()
            log.debug(
                f"   đọc {len(batch)} review, sao={dict(Counter(b['stars'] for b in batch))}"
            )
            total += save_fn(batch)
            if total >= cap:
                return total

            n_before = self.item_count()
            active = self.page.locator(".pagcomment span.active")
            page_before = active.first.inner_text().strip() if active.count() else None

            if self.click_first_visible("load_more"):
                kind = "load_more"
            elif self.click_first_visible("next_page"):
                kind = "next_page"
            else:
                return total  # hết nút để bấm

            try:
                if kind == "load_more":
                    self.page.wait_for_function(
                        "([s, n]) => document.querySelectorAll(s).length > n",
                        arg=[items_sel, n_before],
                        timeout=8000,
                    )
                else:
                    self.page.wait_for_function(
                        "b => { const a = document.querySelector('.pagcomment span.active');"
                        " return a && a.innerText.trim() !== b; }",
                        arg=page_before,
                        timeout=10000,
                    )
                stall = 0
            except PWTimeout:
                stall += 1
                log.debug(f"   {kind}: không thấy thay đổi (stall {stall})")
                if stall >= 2:
                    return total

            sleep()
            self.check_blocked()

        return total


# ----------------------------------------------------------------------------
# CRAWL 1 SẢN PHẨM
# ----------------------------------------------------------------------------
def crawl_product(cr, url, category, seen_ids, out_f, stats):
    product_code = url.split("?")[0].rstrip("/").split("/")[-1]
    log.info(f"\n>> {category} | {product_code}")

    # ------------------------------------------------------------------
    # 1. Mở trang review: ưu tiên /danh-gia, không được thì dùng trang gốc
    # ------------------------------------------------------------------
    source = cr.open_reviews(url)
    if source is None:
        log.warning("   không tìm thấy review ở /danh-gia lẫn trang gốc")
        if DEBUG_MODE:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cr.page.screenshot(path=f"{DEBUG_DIR}/noreview_{product_code}.png")
        return "no_reviews", 0
    log.info(f"   nguồn review: {source}")

    hint = cr.total_hint()
    if hint is not None:
        log.info(f"   tổng đánh giá theo trang: {hint}")

    avg = cr.first_selector("avg_rating")
    if avg:
        try:
            log.info(f"   điểm TB: {cr.page.locator(avg).first.inner_text().strip()}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Hàm lưu 1 lô review (dedupe theo review_id, scrub thông tin cá nhân)
    # ------------------------------------------------------------------
    collected = 0

    def save_batch(batch, star_hint=None):
        nonlocal collected
        added = 0
        for r in batch:
            content = scrub(r.get("content") or "")
            rating = int(r.get("stars") or star_hint or 0)
            if not content or not (1 <= rating <= 5):
                continue

            if star_hint and rating != star_hint:
                log.debug(
                    f"   lệch sao: đọc {rating}★ nhưng đang lọc {star_hint}★ ({r.get('id')})"
                )

            rid = (
                f"{product_code}:{r['id']}"
                if r.get("id")
                else review_id(product_code, rating, r.get("date"), content)
            )
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            date_raw = (r.get("date") or "").strip()
            rec = {
                "review_id": rid,
                "product_code": product_code,
                "category": TARGET_CATEGORIES.get(category, category),
                "category_slug": category,
                "product_url": url,
                "source_page": source,
                "rating": rating,
                "content": content,
                "date_raw": date_raw,
                "date": parse_date(date_raw),
                "crawled_at": datetime.now().isoformat(timespec="seconds"),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[category, rating] += 1
            added += 1
        out_f.flush()
        collected += added
        return added

    # ------------------------------------------------------------------
    # 3. Pha A: lọc riêng 1★, 2★, 3★ (ưu tiên review sao thấp)
    # ------------------------------------------------------------------
    for star in (1, 2, 3):
        if collected >= MAX_PER_PRODUCT:
            break
        if not cr.apply_star_filter(star):
            log.warning(f"   không bật được bộ lọc {star}★")
            continue
        if cr.item_count() == 0:
            log.info(f"   {star}★: không có review")
            continue
        cap = min(PER_STAR_CAP, MAX_PER_PRODUCT - collected)
        n = cr.crawl_pages(cap, lambda b, s=star: save_batch(b, star_hint=s))
        log.info(f"   {star}★: +{n}")

    # ------------------------------------------------------------------
    # 4. Pha B: phần còn lại (kể cả 4-5★) đến khi đủ MAX_PER_PRODUCT
    # ------------------------------------------------------------------
    if collected < MAX_PER_PRODUCT:
        cr.reset_filter()  # trả về False nếu trang không có bộ lọc -> vẫn đọc tiếp
        n = cr.crawl_pages(MAX_PER_PRODUCT - collected, save_batch)
        log.info(f"   tất cả: +{n}")

    if source == "product_page" and hint and collected < hint:
        log.warning(
            f"   trang gốc chỉ lấy được {collected}/{hint} review (không có /danh-gia)"
        )

    log.info(f"   => {collected} review mới")
    return "ok", collected


# ----------------------------------------------------------------------------
# STATE / RESUME
# ----------------------------------------------------------------------------
def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"done": []}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_existing(out_path):
    """Đọc file jsonl đã có để dedupe và đếm tiếp."""
    seen, stats = set(), Counter()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(r["review_id"])
                stats[r["category_slug"], r["rating"]] += 1
    return seen, stats


def print_summary(stats):
    total = sum(stats.values())
    low = sum(v for (c, s), v in stats.items() if s in (1, 2))
    three = sum(v for (c, s), v in stats.items() if s == 3)
    log.info(
        f"\n=== TỔNG: {total} | 1-2★: {low} (cần >=1000) | 3★: {three} (cần >=600) ==="
    )
    cats = Counter()
    for (c, s), v in stats.items():
        cats[c] += v
    for c, v in cats.most_common():
        log.info(f"   {c}: {v}")


# ----------------------------------------------------------------------------
# CHẾ ĐỘ --inspect (bước 4)
# ----------------------------------------------------------------------------
def inspect(url):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        cr = Crawler(page)
        cr.goto(url)
        page.mouse.wheel(0, 3000)
        sleep()
        name = re.sub(r"\W+", "_", urlparse(url).path)[:60]
        with open(f"{DEBUG_DIR}/{name}.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        page.screenshot(path=f"{DEBUG_DIR}/{name}.png", full_page=True)
        print(f"Đã lưu HTML + ảnh vào {DEBUG_DIR}/{name}.*")
        print(
            "Dùng DevTools (F12 > Elements) để xác định selector cho: item, content, date, sao, 'Xem thêm', bộ lọc sao, phần trả lời của ĐMX."
        )
        input("Nhấn Enter để đóng trình duyệt...")
        browser.close()


# ----------------------------------------------------------------------------
# ĐA LUỒNG
# ----------------------------------------------------------------------------
class LockedFile:
    """Bọc file để nhiều thread ghi không bị lẫn dòng."""

    def __init__(self, f):
        self.f = f
        self.lock = threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)

    def flush(self):
        with self.lock:
            self.f.flush()


def merge_stats(dst, src):
    for k, v in src.items():
        dst[k] += v


def worker(wid, q, ctx):
    """1 worker = 1 browser riêng, lấy sản phẩm từ hàng đợi cho đến khi hết / bị chặn / đủ mục tiêu."""
    # xuất phát lệch nhau để không gửi request cùng lúc
    time.sleep(wid * random.uniform(*DELAY))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=ctx.headless)
        try:
            context = browser.new_context(
                viewport={"width": 1280, "height": 900}, locale="vi-VN"
            )
            page = context.new_page()
            cr = Crawler(page, stop=ctx.stop)

            while not ctx.stop.is_set():
                try:
                    url, cat = q.get_nowait()
                except queue.Empty:
                    break

                with ctx.lock:
                    if not ctx.test and totals(ctx.stats)[0] >= TARGET_TOTAL:
                        log.info("Đã đủ số lượng mục tiêu.")
                        break

                local = (
                    Counter()
                )  # thống kê riêng của sản phẩm này, gộp vào chung sau khi xong
                t0 = time.time()
                try:
                    status, n = crawl_product(
                        cr, url, cat, ctx.seen_ids, ctx.out_f, local
                    )
                except BlockedError as e:
                    with ctx.lock:
                        merge_stats(ctx.stats, local)  # review đã ghi vẫn tính
                        first = ctx.block_reason is None
                        if first:
                            ctx.block_reason = (str(e), url)
                    ctx.stop.set()
                    if first:
                        try:
                            os.makedirs(DEBUG_DIR, exist_ok=True)
                            shot = f"{DEBUG_DIR}/blocked_w{wid}_{datetime.now():%Y%m%d_%H%M%S}.png"
                            page.screenshot(path=shot)
                            log.error(f"Đã lưu ảnh trang bị chặn: {shot}")
                        except Exception:
                            pass
                    break
                except Exception as e:
                    # lỗi riêng của 1 sản phẩm: ghi log, KHÔNG đánh dấu done để thử lại lần sau
                    log.error(f"LỖI {url}: {e}")
                    with ctx.lock:
                        merge_stats(ctx.stats, local)
                        ctx.state.setdefault("errors", {})[url] = str(e)
                        save_state(ctx.state_file, ctx.state)
                    continue

                with ctx.lock:
                    merge_stats(ctx.stats, local)
                    ctx.done.add(url)
                    ctx.state["done"] = sorted(ctx.done)
                    ctx.state.setdefault("status", {})[url] = {
                        "result": status,
                        "new": n,
                        "sec": round(time.time() - t0),
                        "worker": wid,
                        "at": datetime.now().isoformat(timespec="seconds"),
                    }
                    ctx.state.get("errors", {}).pop(url, None)
                    save_state(ctx.state_file, ctx.state)
                    ctx.finished += 1
                    total, low, three = totals(ctx.stats)
                    log.info(
                        f"[{ctx.finished}/{ctx.n_todo}] {status.upper()} +{n} | {time.time()-t0:.0f}s | "
                        f"tổng {total} (1-2★ {low}/1000, 3★ {three}/600)"
                    )
                sleep()
        finally:
            browser.close()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inspect",
        metavar="URL",
        help="mở 1 trang /danh-gia, lưu HTML+ảnh để tìm selector",
    )
    ap.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="chạy thử N sản phẩm, ghi ra file *_test_mt",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=3,
        help=f"số worker song song (mặc định 3, tối đa {MAX_WORKERS})",
    )
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--debug", action="store_true", help="hiện log chi tiết trên màn hình"
    )
    ap.add_argument(
        "--refresh-sitemap", action="store_true", help="tải lại sitemap, bỏ qua cache"
    )
    args = ap.parse_args()

    setup_logging(args.debug)

    if args.inspect:
        return inspect(args.inspect)

    workers = max(1, min(args.workers, MAX_WORKERS))
    if workers != args.workers:
        log.warning(f"--workers {args.workers} vượt trần {MAX_WORKERS}; dùng {workers}")

    out_file, state_file = OUT_FILE, STATE_FILE
    if args.test:
        out_file = OUT_FILE.replace(".jsonl", "_test_mt.jsonl")
        state_file = STATE_FILE.replace(".json", "_test_mt.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    state = load_state(state_file)
    done = set(state["done"])
    seen_ids, stats = load_existing(out_file)

    log.info("Đang lấy danh sách sản phẩm từ sitemap...")
    products = prioritize(get_product_urls(refresh=args.refresh_sitemap))
    log.info(
        f"Tổng {len(products)} sản phẩm thuộc ngành hàng mục tiêu; đã crawl {len(done)}."
    )
    todo = [(u, c) for u, c in products if u not in done]
    if args.test:
        todo = todo[: args.test]
    log.info(
        f"Cần crawl trong lần chạy này: {len(todo)} sản phẩm | {workers} worker | "
        f"mỗi worker nghỉ {DELAY[0]}-{DELAY[1]}s"
    )

    q = queue.Queue()
    for item in todo:
        q.put(item)

    t_start = time.time()
    with open(out_file, "a", encoding="utf-8") as raw_f:
        ctx = SimpleNamespace(
            stop=threading.Event(),
            lock=threading.Lock(),
            out_f=LockedFile(raw_f),
            seen_ids=seen_ids,
            stats=stats,
            state=state,
            state_file=state_file,
            done=done,
            headless=args.headless,
            test=bool(args.test),
            n_todo=len(todo),
            finished=0,
            block_reason=None,
        )
        threads = [
            threading.Thread(target=worker, args=(i, q, ctx), name=f"w{i}")
            for i in range(workers)
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            log.warning("Ctrl+C -> yêu cầu các worker dừng sau thao tác hiện tại...")
            ctx.stop.set()
            for t in threads:
                t.join()

    elapsed = time.time() - t_start
    log.info(
        f"Xong {ctx.finished}/{len(todo)} sản phẩm trong {elapsed/60:.1f} phút "
        f"({workers} worker) | {totals(stats)[0]} review"
    )

    exit_code = 0
    if ctx.block_reason:
        reason, url = ctx.block_reason
        log.error(f"DỪNG: {reason} (worker đầu tiên gặp chặn ở {url})")
        log.error(
            "Hãy báo cho nhóm. Dữ liệu và tiến độ đã được lưu; chạy lại sau để tiếp tục."
        )
        exit_code = 2

    print_summary(stats)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
