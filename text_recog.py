import re
import csv
from pathlib import Path

from pypdf import mult

# Regexes for date & time
day_re   = re.compile(r"^(0?[1-9]|[12][0-9]|3[01])$")
year_re  = re.compile(r"^(19|20)\d{2}$")
month_re = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)$"
)  
full_date_re = re.compile(
    r"^(0?[1-9]|[12][0-9]|3[01])\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(19|20)\d{2}$"
)
day_month_re   = re.compile( 
    r"^(0?[1-9]|[12][0-9]|3[01])\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)$"
)
month_year_re  = re.compile(  
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(19|20)\d{2}$"
)

time_full_re = re.compile(r"^(?:0?[1-9]|1[0-2])[:.][0-5]\d\s*[AaPp][Mm]$")
time_core_re = re.compile(r"^(?:0?[1-9]|1[0-2])[:.][0-5]\d$")
ampm_re      = re.compile(r"^[AaPp][Mm]$")


def center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(sum(xs)) / 4.0, float(sum(ys)) / 4.0

def same_line(b1, b2, y_thresh=10):
    _, y1 = center(b1); _, y2 = center(b2)
    return abs(y1 - y2) <= y_thresh

def below(b1, b2, y_thresh=10):
    _, y1 = center(b1); _, y2 = center(b2)
    return (y2 - y1) > y_thresh

def sort_reading_order(dets):

    return sorted(dets, key=lambda d: (center(d[0])[1], center(d[0])[0]))  


# Date Extraction 

def normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())

def extract_date_with_lookahead(i, dets):
    """
    Try to read a date starting at dets[i].
    Returns (date_str, next_index) or (None, i).

    """
    bbox1, text1, _ = dets[i]
    t1 = normalize_spaces(text1)


    if full_date_re.fullmatch(t1):
        return t1, i + 1

    if day_month_re.fullmatch(t1) and i + 1 < len(dets):
        bbox2, text2, _ = dets[i + 1]
        t2 = normalize_spaces(text2)
        if same_line(bbox1, bbox2) and year_re.fullmatch(t2):
            return f"{t1} {t2}", i + 2

    if day_re.fullmatch(t1) and i + 1 < len(dets):
        bbox2, text2, _ = dets[i + 1]
        t2 = normalize_spaces(text2)
        if same_line(bbox1, bbox2) and month_year_re.fullmatch(t2):
            return f"{t1} {t2}", i + 2

    if day_re.fullmatch(t1) and i + 2 < len(dets):
        bbox2, text2, _ = dets[i + 1]
        bbox3, text3, _ = dets[i + 2]
        t2 = normalize_spaces(text2)
        t3 = normalize_spaces(text3)
        if same_line(bbox1, bbox2) and same_line(bbox1, bbox3):
            if month_re.fullmatch(t2) and year_re.fullmatch(t3):
                return f"{t1} {t2} {t3}", i + 3

    return None, i

# Time extraction

def normalize_time(raw: str) -> str:
    s = raw.strip().replace(" ", "")
    s = s.replace(".", ":")
    m = re.match(r"^(.+?)([AaPp][Mm])?$", s)
    core = m.group(1)
    suf = m.group(2)
    if suf:
        return f"{core} {suf.upper()}"
    return core

def extract_time_with_lookahead(i, dets):
    """
    If det[i] (and optionally det[i+1]) form a time, return
    (time_str, next_index, time_bbox); else (None, i, None).

    """
    bbox, text, _ = dets[i]
    raw = text.strip()

    compact = raw.replace(" ", "").replace(".", ":")
    if time_full_re.fullmatch(compact):
        return normalize_time(compact), i + 1, bbox

    core_candidate = raw.replace(" ", "").replace(".", ":")
    if not time_core_re.fullmatch(core_candidate):
        return None, i, None

    core_norm = normalize_time(raw)

    if i + 1 < len(dets):
        bbox2, text2, _ = dets[i + 1]
        t2 = text2.strip()
        if ampm_re.fullmatch(t2) and same_line(bbox, bbox2):
            return f"{core_norm} {t2.upper()}", i + 2, bbox

    return core_norm, i + 1, bbox

# Group Messages by timestamp 

def sender_from_timebox(time_bbox, img_width, multi = 0.8):
    xs = [p[0] for p in time_bbox]
    min_x, max_x = min(xs), max(xs)
    if img_width * multi <= max_x:
        return "B"
    return "A"

def messages_from_ocr(easyocr_result, image_name, img_height, img_width):
    
    dets = sort_reading_order(easyocr_result)

    rows = []
    current_date = ""
    current_time = None
    current_sender = None
    current_msg_parts = []

    header_cut_y = img_height * 0.12  

    i = 0
    n = len(dets)

    while i < n:
        bbox, text, conf = dets[i]
        # print(bbox, text)
        text = text.strip()
        _, cy = center(bbox)

        if cy < header_cut_y:
            i += 1
            continue

        # 1) Date bubble
        date_str, next_i = extract_date_with_lookahead(i, dets)
        if date_str:
            current_date = date_str
            i = next_i
            continue

        # 2) Time token
        time_str, j, time_bbox = extract_time_with_lookahead(i, dets)
        if time_str:
            is_timestamp = True
            if j < n:
                next_bbox, _, _ = dets[j]
                if not below(time_bbox, next_bbox):
                    is_timestamp = False
                    current_msg_parts.append(time_str)

            if is_timestamp:
                # close previous message if any
                current_sender = sender_from_timebox(time_bbox, img_width)
                if current_msg_parts:
                    rows.append({
                        "Date": current_date,
                        "Time": time_str,
                        "Sender": current_sender,
                        "Message": " ".join(current_msg_parts),
                        "Source": image_name,
                    })
                    current_msg_parts = []
                    
            i = j
            continue

        # 3) Skip bottom UI
        if text.lower().startswith("type a message"):
            i += 1
            continue

        # 4) Regular message text
        current_msg_parts.append(text)
        i += 1

    if current_msg_parts:
        rows.append({
            "Date": current_date,
            "Time": current_time or "",
            "Sender": current_sender or "INCOMING",
            "Message": " ".join(current_msg_parts),
            "Source": image_name,
        })

    return rows


# if __name__ == "__main__":
#     import easyocr
#     from PIL import Image

#     image_path = "Screenshot 1.png"
#     img = Image.open(image_path)
#     img_width, img_height = img.size
#     print(f"Image size: {img_width}x{img_height}")
#     reader = easyocr.Reader(["en"])
#     easyocr_result = reader.readtext(image_path, detail=1)

#     rows = messages_from_ocr(easyocr_result, Path(image_path).name, img_height, img_width)

#     with open("chat_output.csv", "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=["Date", "Time", "Sender", "Message", "Source"])
#         writer.writeheader()
#         writer.writerows(rows)
