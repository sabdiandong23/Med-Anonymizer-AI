import os
import json
import math
import random
import shutil
from datetime import datetime, timedelta

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================================================
# 基本配置
# =========================================================
TARGET_SIZE = (1024, 1024)
MARGIN = 18
PLACEMENT_PAD = 8
MAX_PLACEMENT_TRIES = 100

# 真 PHI / 假 PHI 都更偏向四角，但不是固定死
REAL_CORNER_BIAS = 0.75
FAKE_CORNER_BIAS = 0.70

# 每个原始 DICOM 生成几个样本
SAMPLES_PER_DCM = 1

# 每个 split 中 20% 生成“纯假 PHI 负样本”
FAKE_ONLY_RATIO = 0.20

# 是否清空旧的 train/val/test 再重建
CLEAR_OLD_OUTPUT = True

SAVE_META_JSON = True

# 字号 / 描边
REAL_FONT_SIZE_RANGE = (18, 36)
FAKE_FONT_SIZE_RANGE = (18, 34)
STROKE_WIDTH_CHOICES = [1, 1, 2]

# 颜色：按你的要求，白 / 黄
REAL_COLORS = [
    (255, 255, 255),
    (255, 255, 0),
]

# 假 PHI 大部分也用白/黄，少部分用浅灰等，增加迷惑性
FAKE_COLORS = [
    (255, 255, 255),
    (255, 255, 0),
    (210, 210, 210),
    (180, 180, 180),
    (220, 220, 160),
    (170, 210, 210),
]

# 模糊
TEXT_BLUR_PROB = 0.35
TEXT_BLUR_RADIUS_RANGE = (0.4, 1.2)

IMAGE_BLUR_PROB = 0.15
IMAGE_BLUR_RADIUS_RANGE = (0.2, 0.8)

# 假 PHI 数量
REAL_SAMPLE_FAKE_COUNT_RANGE = (5, 10)   # 正常样本中附带的假PHI数量
FAKE_ONLY_COUNT_RANGE = (8, 14)          # 纯负样本中假PHI数量

# 少量允许假 PHI 轻微靠近/重叠
FAKE_OVERLAP_PROB = 0.20

# 类别映射
CLASS_MAP = {
    "patient": 0,
    "time": 1,
    "institution": 2
}

CLASS_NAMES = [
    "patient_info",
    "time_info",
    "institution_info"
]


# =========================================================
# 字体池：一个集一个集生成
# =========================================================
TRAIN_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]

VAL_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
]

TEST_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\fangsong.ttf",
    r"C:\Windows\Fonts\DENG.TTF",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\trebuc.ttf",
    r"C:\Windows\Fonts\cour.ttf",
]


# =========================================================
# 随机数据池
# =========================================================
CN_SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜")
CN_GIVEN_1 = list("伟刚勇毅俊峰强军平斌辉明永健世广志义兴良海山仁波宁贵福生龙元全国胜学祥才发武新利清飞彬富顺信子杰涛昌成康星光天达安岩中茂进林有坚和彪博诚先敬震振壮会思群豪心邦承乐绍功松善厚庆磊民友裕河哲江超浩亮政谦亨奇固之轮翰朗伯宏言若鸣朋斋梁栋维启克伦翔旭鹏泽晨辰士以建家致树炎德行时泰盛雄琛钧冠策腾榕风航弘")
CN_GIVEN_2 = list("华明强军涛洋斌玲娜静敏丽艳娟磊鹏宇晨浩瑞凯杰鑫龙辉琳雪婷倩涵颖怡琪彤瑶珊瑜博浩峰然宁悦")

EN_FIRST = ["James", "Emma", "Liam", "Olivia", "Noah", "Sophia", "Mason", "Ava", "Ethan", "Isabella", "William", "Mia"]
EN_LAST = ["Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Lee", "Walker"]

CN_HOSPITALS = ["华东医院", "市第一人民医院", "仁和医院", "中山医学中心", "协和影像中心", "滨海医院"]
EN_HOSPITALS = ["Central City Hospital", "Union Medical Center", "Riverside Imaging Center", "Saint Mary Hospital", "Metro General Hospital"]

CN_STREETS = ["中山路", "人民路", "解放路", "和平路", "建设路", "新华路", "长江路", "延安路"]
CN_CITIES = ["上海", "北京", "广州", "深圳", "杭州", "南京", "武汉", "成都"]

EN_STREETS = ["Main St", "Broadway", "Maple Ave", "Oak St", "Pine Rd", "6th Ave", "Park Blvd"]
EN_CITIES = ["Boston", "Seattle", "Austin", "Chicago", "San Diego", "Houston", "Phoenix"]


# =========================================================
# 工具函数
# =========================================================
def load_random_font(size: int, font_candidates):
    available = [fp for fp in font_candidates if os.path.exists(fp)]
    if not available:
        return ImageFont.load_default(), "default"
    font_path = random.choice(available)
    return ImageFont.truetype(font_path, size), font_path


def rand_dt(start_year=2021, end_year=2025):
    start = datetime(start_year, 1, 1, 0, 0, 0)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta))


def boxes_overlap(box1, box2, pad=0):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return not (
        x2 + pad < a1 or
        a2 + pad < x1 or
        y2 + pad < b1 or
        b2 + pad < y1
    )


def sample_random_xy_with_bias(text_w, text_h, img_w, img_h, corner_bias=0.5):
    max_x = max(MARGIN, img_w - text_w - MARGIN)
    max_y = max(MARGIN, img_h - text_h - MARGIN)

    if random.random() < corner_bias:
        corner = random.choice(["tl", "tr", "bl", "br"])
        x_band = max(80, img_w // 5)
        y_band = max(80, img_h // 5)

        if corner == "tl":
            x = random.randint(MARGIN, min(max_x, MARGIN + x_band))
            y = random.randint(MARGIN, min(max_y, MARGIN + y_band))
        elif corner == "tr":
            x = random.randint(max(MARGIN, max_x - x_band), max_x)
            y = random.randint(MARGIN, min(max_y, MARGIN + y_band))
        elif corner == "bl":
            x = random.randint(MARGIN, min(max_x, MARGIN + x_band))
            y = random.randint(max(MARGIN, max_y - y_band), max_y)
        else:
            x = random.randint(max(MARGIN, max_x - x_band), max_x)
            y = random.randint(max(MARGIN, max_y - y_band), max_y)
    else:
        x = random.randint(MARGIN, max_x)
        y = random.randint(MARGIN, max_y)

    return x, y


def find_non_overlapping_position(text_w, text_h, img_w, img_h, occupied_boxes, corner_bias=0.5):
    for _ in range(MAX_PLACEMENT_TRIES):
        x, y = sample_random_xy_with_bias(text_w, text_h, img_w, img_h, corner_bias)
        candidate = (x, y, x + text_w, y + text_h)
        if not any(boxes_overlap(candidate, ob, pad=PLACEMENT_PAD) for ob in occupied_boxes):
            return x, y, candidate

    x, y = sample_random_xy_with_bias(text_w, text_h, img_w, img_h, corner_bias)
    candidate = (x, y, x + text_w, y + text_h)
    return x, y, candidate


def find_position_for_fake(text_w, text_h, img_w, img_h, occupied_boxes):
    allow_overlap = random.random() < FAKE_OVERLAP_PROB

    for _ in range(MAX_PLACEMENT_TRIES):
        x, y = sample_random_xy_with_bias(text_w, text_h, img_w, img_h, FAKE_CORNER_BIAS)
        candidate = (x, y, x + text_w, y + text_h)

        if allow_overlap:
            return x, y, candidate

        if not any(boxes_overlap(candidate, ob, pad=PLACEMENT_PAD) for ob in occupied_boxes):
            return x, y, candidate

    x, y = sample_random_xy_with_bias(text_w, text_h, img_w, img_h, FAKE_CORNER_BIAS)
    candidate = (x, y, x + text_w, y + text_h)
    return x, y, candidate


def make_safe_stem(file_path, root_folder):
    rel = os.path.relpath(file_path, root_folder)
    stem = os.path.splitext(rel)[0]
    stem = stem.replace("\\", "_").replace("/", "_")
    return stem


def get_unique_output_stem(out_img_dir, base_stem):
    stem = base_stem
    idx = 1
    while os.path.exists(os.path.join(out_img_dir, stem + ".png")):
        stem = f"{base_stem}_{idx}"
        idx += 1
    return stem


# =========================================================
# 真 PHI 生成函数
# =========================================================
def gen_cn_name():
    return random.choice(CN_SURNAMES) + random.choice(CN_GIVEN_1) + random.choice(CN_GIVEN_2)


def gen_en_name():
    return f"{random.choice(EN_LAST).upper()}^{random.choice(EN_FIRST).upper()}"


def gen_person_name(lang):
    return gen_cn_name() if lang == "zh" else gen_en_name()


def gen_patient_id(lang):
    if random.random() < 0.5:
        return f"P{random.randint(100000, 999999)}"
    return f"{random.randint(10000000, 99999999)}"


def gen_sex(lang):
    if lang == "zh":
        return random.choice(["男", "女"])
    return random.choice(["M", "F", "Male", "Female"])


def gen_birth_date(lang):
    dt = rand_dt(1940, 2012)
    if lang == "zh":
        return random.choice([
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%Y/%m/%d"),
            dt.strftime("%Y年%m月%d日")
        ])
    return random.choice([
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%m/%d/%Y"),
        dt.strftime("%Y%m%d")
    ])


def gen_study_time(lang):
    dt = rand_dt(2021, 2025)
    if lang == "zh":
        return random.choice([
            dt.strftime("%Y-%m-%d %H:%M"),
            dt.strftime("%Y/%m/%d %H:%M:%S"),
            dt.strftime("%Y年%m月%d日 %H:%M")
        ])
    return random.choice([
        dt.strftime("%Y-%m-%d %H:%M"),
        dt.strftime("%Y%m%d %H%M%S"),
        dt.strftime("%m/%d/%Y %H:%M")
    ])


def gen_visit_time(lang):
    dt = rand_dt(2021, 2025)
    if lang == "zh":
        return random.choice([
            dt.strftime("%Y-%m-%d %H:%M"),
            dt.strftime("%Y/%m/%d %H:%M:%S"),
        ])
    return random.choice([
        dt.strftime("%Y-%m-%d %H:%M"),
        dt.strftime("%m/%d/%Y %H:%M"),
    ])


def gen_study_date(lang):
    dt = rand_dt(2021, 2025)
    if lang == "zh":
        return random.choice([
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%Y/%m/%d"),
            dt.strftime("%Y年%m月%d日")
        ])
    return random.choice([
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%m/%d/%Y"),
        dt.strftime("%Y%m%d")
    ])


def gen_institution_name(lang):
    return random.choice(CN_HOSPITALS) if lang == "zh" else random.choice(EN_HOSPITALS)


def gen_institution_address(lang):
    if lang == "zh":
        city = random.choice(CN_CITIES)
        street = random.choice(CN_STREETS)
        no = random.randint(10, 1999)
        return f"{city}市{street}{no}号"
    city = random.choice(EN_CITIES)
    street = random.choice(EN_STREETS)
    no = random.randint(10, 1999)
    return f"{no} {street}, {city}"


def make_real_entry(key, category, zh_label, en_label, value_fn):
    lang = random.choice(["zh", "en"])
    label = zh_label if lang == "zh" else en_label
    sep = "：" if lang == "zh" else ": "
    value = value_fn(lang)
    return {
        "key": key,
        "category": category,
        "class_id": CLASS_MAP[category],
        "label": label,
        "sep": sep,
        "value": value,
        "lang": lang,
    }


def build_real_entries():
    # 这里把你现在定义的所有真 PHI 都放进来
    entries = [
        make_real_entry("patient_name", "patient", "患者姓名", "Patient Name", gen_person_name),
        make_real_entry("patient_id", "patient", "患者ID", "Patient ID", gen_patient_id),
        make_real_entry("patient_sex", "patient", "性别", "Sex", gen_sex),
        make_real_entry("birth_date", "patient", "出生日期", "Date of Birth", gen_birth_date),

        make_real_entry("study_time", "time", "拍片时间", "Study Time", gen_study_time),
        make_real_entry("visit_time", "time", "就诊时间", "Visit Time", gen_visit_time),
        make_real_entry("study_date", "time", "检查日期", "Study Date", gen_study_date),

        make_real_entry("operator_name", "institution", "操作医生", "Operator Name", gen_person_name),
        make_real_entry("referring_physician", "institution", "问诊医生", "Referring Physician", gen_person_name),
        make_real_entry("institution_name", "institution", "机构名称", "Institution Name", gen_institution_name),
        make_real_entry("institution_address", "institution", "机构地址", "Institution Address", gen_institution_address),
    ]
    random.shuffle(entries)
    return entries


# =========================================================
# 假 PHI 生成函数（不标注）
# =========================================================
def gen_patient_position(lang):
    if lang == "zh":
        return random.choice(["仰卧位", "俯卧位", "头先进仰卧位", "足先进仰卧位"])
    return random.choice(["HFS", "FFS", "Supine", "Prone"])


def gen_body_part(lang):
    if lang == "zh":
        return random.choice(["胸部", "腹部", "头部", "颈部", "胸腹部", "盆腔"])
    return random.choice(["Chest", "Abdomen", "Head", "Neck", "Chest-Abdomen", "Pelvis"])


def gen_orientation(lang):
    if lang == "zh":
        return random.choice(["横断位", "冠状位", "矢状位", "斜位"])
    return random.choice(["Axial", "Coronal", "Sagittal", "Oblique"])


def gen_protocol_name(lang):
    if lang == "zh":
        return random.choice(["胸部平扫", "增强扫描", "常规重建", "薄层重建", "定位像"])
    return random.choice(["Chest Routine", "Contrast Study", "Standard Recon", "Thin Recon", "Scout"])


def gen_series_number(lang):
    return str(random.randint(1, 12))


def gen_image_number(lang):
    return str(random.randint(1, 180))


def gen_slice_thickness(lang):
    return f"{random.choice([1.0, 2.5, 5.0, 7.5])}mm"


def gen_window_width(lang):
    return str(random.randint(200, 1200))


def gen_window_level(lang):
    return str(random.randint(-200, 200))


def gen_matrix_size(lang):
    return random.choice(["256x256", "320x320", "512x512", "1024x1024"])


def gen_kvp(lang):
    return str(random.choice([80, 100, 120, 140]))


def gen_mas(lang):
    return str(random.randint(80, 320))


def gen_recon_kernel(lang):
    if lang == "zh":
        return random.choice(["标准", "软组织", "骨算法", "肺算法"])
    return random.choice(["STANDARD", "SOFT", "BONE", "LUNG"])


def gen_study_desc(lang):
    if lang == "zh":
        return random.choice(["胸部CT", "腹部CT", "头颅MR", "盆腔MR", "全身PET"])
    return random.choice(["Chest CT", "Abdomen CT", "Brain MR", "Pelvis MR", "Whole-body PET"])


def make_fake_entry(zh_label, en_label, value_fn):
    lang = random.choice(["zh", "en"])
    label = zh_label if lang == "zh" else en_label
    sep = "：" if lang == "zh" else ": "
    value = value_fn(lang)
    return {
        "label": label,
        "sep": sep,
        "value": value,
        "lang": lang
    }


def build_fake_entries(count_range):
    pool = [
        make_fake_entry("患者体位", "Patient Position", gen_patient_position),
        make_fake_entry("检查部位", "Body Part", gen_body_part),
        make_fake_entry("图像方向", "Orientation", gen_orientation),
        make_fake_entry("扫描协议", "Protocol Name", gen_protocol_name),
        make_fake_entry("序列编号", "Series Number", gen_series_number),
        make_fake_entry("图像编号", "Image Number", gen_image_number),
        make_fake_entry("层厚", "Slice Thickness", gen_slice_thickness),
        make_fake_entry("窗宽", "Window Width", gen_window_width),
        make_fake_entry("窗位", "Window Level", gen_window_level),
        make_fake_entry("矩阵大小", "Matrix", gen_matrix_size),
        make_fake_entry("管电压", "KVP", gen_kvp),
        make_fake_entry("管电流", "mAs", gen_mas),
        make_fake_entry("重建核", "Recon Kernel", gen_recon_kernel),
        make_fake_entry("检查描述", "Study Description", gen_study_desc),
    ]

    k = random.randint(*count_range)
    k = min(k, len(pool))
    selected = random.sample(pool, k)
    random.shuffle(selected)
    return selected


# =========================================================
# DICOM 预处理
# =========================================================
def preprocess_dicom_to_rgb(ds):
    pixel = ds.pixel_array

    if pixel.ndim == 3:
        if pixel.shape[-1] == 3:
            arr = pixel
        else:
            arr = pixel[0]
    else:
        arr = pixel

    arr = arr.astype(np.float32)

    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1 + 1e-6)
    arr = (arr * 255).astype(np.uint8)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr

    img = Image.fromarray(arr).resize(TARGET_SIZE, Image.Resampling.BILINEAR).convert("RGB")
    return img


# =========================================================
# 绘制函数
# =========================================================
def draw_text_with_optional_blur(
    base_img, x, y, text, font, fill,
    stroke_width=1, stroke_fill=(0, 0, 0),
    blur_prob=None, blur_radius_range=None
):
    if blur_prob is None:
        blur_prob = TEXT_BLUR_PROB
    if blur_radius_range is None:
        blur_radius_range = TEXT_BLUR_RADIUS_RANGE

    dummy_draw = ImageDraw.Draw(base_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)

    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    pad = 6
    layer = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    rgba_fill = fill + (255,)
    rgba_stroke = stroke_fill + (255,)

    d.text(
        (pad, pad),
        text,
        font=font,
        fill=rgba_fill,
        stroke_width=stroke_width,
        stroke_fill=rgba_stroke
    )

    if random.random() < blur_prob:
        radius = random.uniform(*blur_radius_range)
        layer = layer.filter(ImageFilter.GaussianBlur(radius=radius))

    base_img.alpha_composite(layer, (x - pad, y - pad))


def draw_real_entries(img, real_entries, font_candidates):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    ann_records = []
    occupied_boxes = []

    for e in real_entries:
        full_text = e["label"] + e["sep"] + e["value"]

        font_size = random.randint(*REAL_FONT_SIZE_RANGE)
        font, font_path = load_random_font(font_size, font_candidates)
        stroke_width = random.choice(STROKE_WIDTH_CHOICES)
        color = random.choice(REAL_COLORS)

        full_box = draw.textbbox((0, 0), full_text, font=font, stroke_width=stroke_width)
        text_w = full_box[2] - full_box[0]
        text_h = full_box[3] - full_box[1]

        x, y, candidate = find_non_overlapping_position(
            text_w, text_h, w, h, occupied_boxes, REAL_CORNER_BIAS
        )
        occupied_boxes.append(candidate)

        draw_text_with_optional_blur(
            img, x, y, full_text,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0)
        )

        bx1 = x + full_box[0]
        by1 = y + full_box[1]
        bx2 = x + full_box[2]
        by2 = y + full_box[3]

        ann_records.append({
            "key": e["key"],
            "category": e["category"],
            "class_id": e["class_id"],
            "label_text": e["label"],
            "value_text": e["value"],
            "full_text": full_text,
            "lang": e["lang"],
            "font_size": font_size,
            "font_path": font_path,
            "bbox_xyxy": [int(bx1), int(by1), int(bx2), int(by2)]
        })

    return img.convert("RGB"), ann_records, occupied_boxes


def draw_fake_entries(img, fake_entries, font_candidates, occupied_boxes):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    fake_records = []

    for e in fake_entries:
        full_text = e["label"] + e["sep"] + e["value"]

        font_size = random.randint(*FAKE_FONT_SIZE_RANGE)
        font, font_path = load_random_font(font_size, font_candidates)
        stroke_width = random.choice(STROKE_WIDTH_CHOICES)
        color = random.choice(FAKE_COLORS)

        full_box = draw.textbbox((0, 0), full_text, font=font, stroke_width=stroke_width)
        text_w = full_box[2] - full_box[0]
        text_h = full_box[3] - full_box[1]

        x, y, candidate = find_position_for_fake(text_w, text_h, w, h, occupied_boxes)
        occupied_boxes.append(candidate)

        draw_text_with_optional_blur(
            img, x, y, full_text,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0)
        )

        fake_records.append({
            "label_text": e["label"],
            "value_text": e["value"],
            "full_text": full_text,
            "lang": e["lang"],
            "font_size": font_size,
            "font_path": font_path,
            "bbox_xyxy": [int(x + full_box[0]), int(y + full_box[1]), int(x + full_box[2]), int(y + full_box[3])]
        })

    return img.convert("RGB"), fake_records


def synthesize_normal_sample(img, font_candidates):
    # 正常样本：真 PHI + 假 PHI
    real_entries = build_real_entries()
    img, ann_records, occupied_boxes = draw_real_entries(img, real_entries, font_candidates)

    fake_entries = build_fake_entries(REAL_SAMPLE_FAKE_COUNT_RANGE)
    img, fake_records = draw_fake_entries(img, fake_entries, font_candidates, occupied_boxes)

    if random.random() < IMAGE_BLUR_PROB:
        radius = random.uniform(*IMAGE_BLUR_RADIUS_RANGE)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img.convert("RGB"), ann_records, fake_records


def synthesize_fake_only_sample(img, font_candidates):
    # 纯负样本：只有假 PHI，没有真 PHI
    occupied_boxes = []
    fake_entries = build_fake_entries(FAKE_ONLY_COUNT_RANGE)
    img, fake_records = draw_fake_entries(img, fake_entries, font_candidates, occupied_boxes)

    if random.random() < IMAGE_BLUR_PROB:
        radius = random.uniform(*IMAGE_BLUR_RADIUS_RANGE)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    ann_records = []
    return img.convert("RGB"), ann_records, fake_records


# =========================================================
# 保存
# =========================================================
def save_yolo_label(label_path, ann_records, img_w, img_h):
    with open(label_path, "w", encoding="utf-8") as f:
        if not ann_records:
            return

        for rec in ann_records:
            x1, y1, x2, y2 = rec["bbox_xyxy"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            xc = (x1 + x2) / 2.0 / img_w
            yc = (y1 + y2) / 2.0 / img_h
            nw = bw / img_w
            nh = bh / img_h
            f.write(f"{rec['class_id']} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")


def save_meta_json(meta_path, image_name, mode, ann_records, fake_records):
    obj = {
        "image_name": image_name,
        "mode": mode,  # "normal" 或 "fake_only"
        "class_names": CLASS_NAMES,
        "records": ann_records,
        "fake_phi_records": fake_records
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_dataset_files(output_root):
    with open(os.path.join(output_root, "classes.txt"), "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")

    hide_policy = {
        "patient_info": "hide",
        "time_info": "hide",
        "institution_info": "hide"
    }
    with open(os.path.join(output_root, "hide_policy_example.json"), "w", encoding="utf-8") as f:
        json.dump(hide_policy, f, ensure_ascii=False, indent=2)

    data_yaml = f"""path: {output_root}
train: train/images
val: val/images
test: test/images

names:
  0: patient_info
  1: time_info
  2: institution_info
"""
    with open(os.path.join(output_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)


# =========================================================
# 单张处理
# =========================================================
def process_dicom(
    dcm_path,
    input_root,
    out_img_dir,
    out_label_dir,
    out_meta_dir,
    font_candidates,
    mode,
    variant_idx=0
):
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        img = preprocess_dicom_to_rgb(ds)

        if mode == "normal":
            img, ann_records, fake_records = synthesize_normal_sample(img, font_candidates)
        elif mode == "fake_only":
            img, ann_records, fake_records = synthesize_fake_only_sample(img, font_candidates)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        stem = make_safe_stem(dcm_path, input_root)
        stem = f"{mode}_{stem}"
        if variant_idx > 0:
            stem = f"{stem}_aug{variant_idx}"
        stem = get_unique_output_stem(out_img_dir, stem)

        img.save(os.path.join(out_img_dir, stem + ".png"))

        save_yolo_label(
            os.path.join(out_label_dir, stem + ".txt"),
            ann_records,
            img.width,
            img.height
        )

        if SAVE_META_JSON:
            save_meta_json(
                os.path.join(out_meta_dir, stem + ".json"),
                stem + ".png",
                mode,
                ann_records,
                fake_records
            )

        print(f"[OK] {mode}: {stem}")

    except Exception as e:
        print(f"[ERROR] {dcm_path} -> {e}")


# =========================================================
# split 生成
# =========================================================
def prepare_output_split(split_output_root):
    out_img_dir = os.path.join(split_output_root, "images")
    out_label_dir = os.path.join(split_output_root, "labels")
    out_meta_dir = os.path.join(split_output_root, "meta")

    if CLEAR_OLD_OUTPUT and os.path.exists(split_output_root):
        shutil.rmtree(split_output_root)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    return out_img_dir, out_label_dir, out_meta_dir


def generate_split(input_folder, split_output_root, font_candidates):
    out_img_dir, out_label_dir, out_meta_dir = prepare_output_split(split_output_root)

    dcm_files = []
    for root, _, files in os.walk(input_folder):
        for fn in files:
            if fn.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, fn))

    # 扩展成任务列表，这样能精确控制 80% / 20%
    tasks = []
    for dcm_path in dcm_files:
        for i in range(SAMPLES_PER_DCM):
            tasks.append((dcm_path, i))

    total = len(tasks)
    num_fake_only = int(round(total * FAKE_ONLY_RATIO))
    num_normal = total - num_fake_only

    modes = ["normal"] * num_normal + ["fake_only"] * num_fake_only
    random.shuffle(modes)
    random.shuffle(tasks)

    print(f"\n[Split] {split_output_root}")
    print(f"Total DICOM files: {len(dcm_files)}")
    print(f"Total samples: {total}")
    print(f"Normal samples (80%): {num_normal}")
    print(f"Fake-only samples (20%): {num_fake_only}")

    for (dcm_path, variant_idx), mode in zip(tasks, modes):
        process_dicom(
            dcm_path=dcm_path,
            input_root=input_folder,
            out_img_dir=out_img_dir,
            out_label_dir=out_label_dir,
            out_meta_dir=out_meta_dir,
            font_candidates=font_candidates,
            mode=mode,
            variant_idx=variant_idx
        )


# =========================================================
# 主函数
# =========================================================
if __name__ == "__main__":
    output_root = r"F:\数据集\数据集11111"

    train_input = r"F:\数据集\数据集11111\训练集"
    val_input   = r"F:\数据集\数据集11111\验证集"
    test_input  = r"F:\数据集\数据集11111\测试集"

    save_dataset_files(output_root)

    generate_split(
        input_folder=train_input,
        split_output_root=os.path.join(output_root, "train"),
        font_candidates=TRAIN_FONT_CANDIDATES
    )

    generate_split(
        input_folder=val_input,
        split_output_root=os.path.join(output_root, "val"),
        font_candidates=VAL_FONT_CANDIDATES
    )

    generate_split(
        input_folder=test_input,
        split_output_root=os.path.join(output_root, "test"),
        font_candidates=TEST_FONT_CANDIDATES
    )



# import os
# import json
# import random
# from datetime import datetime, timedelta
#
# import numpy as np
# import pydicom
# from PIL import Image, ImageDraw, ImageFont, ImageFilter
#
#
# # =========================================================
# # 目标：
# # 只生成“假 PHI 负样本”，并追加到现有 YOLO 数据集
# # 输出：
# #   train/images, train/labels, train/meta
# #   val/images,   val/labels,   val/meta
# #   test/images,  test/labels,  test/meta
# #
# # 特点：
# # - 图片里只有高仿 PHI 干扰
# # - 没有真实 PHI
# # - labels/*.txt 为空（负样本）
# # - 不覆盖现有数据，自动避免重名
# # =========================================================
#
#
# # =========================================================
# # 基本配置
# # =========================================================
# TARGET_SIZE = (1024, 1024)
# MARGIN = 18
# PLACEMENT_PAD = 8
# MAX_PLACEMENT_TRIES = 100
#
# # 假 PHI 大多靠近四角，但不是固定死
# DECOY_CORNER_BIAS = 0.70
#
# # 每张底图生成几个负样本版本
# NEG_PER_DCM = 1
#
# # 输出文件名前缀，避免和已有正样本撞名
# NEGATIVE_PREFIX = "negfake"
#
# # =========================================================
# # 字体池：train / val / test 可以不同
# # =========================================================
# TRAIN_FONT_CANDIDATES = [
#     r"C:\Windows\Fonts\msyh.ttc",
#     r"C:\Windows\Fonts\msyhbd.ttc",
#     r"C:\Windows\Fonts\simhei.ttf",
#     r"C:\Windows\Fonts\arial.ttf",
#     r"C:\Windows\Fonts\calibri.ttf",
# ]
#
# VAL_FONT_CANDIDATES = [
#     r"C:\Windows\Fonts\simsun.ttc",
#     r"C:\Windows\Fonts\simkai.ttf",
#     r"C:\Windows\Fonts\verdana.ttf",
#     r"C:\Windows\Fonts\segoeui.ttf",
#     r"C:\Windows\Fonts\georgia.ttf",
# ]
#
# TEST_FONT_CANDIDATES = [
#     r"C:\Windows\Fonts\fangsong.ttf",
#     r"C:\Windows\Fonts\DENG.TTF",
#     r"C:\Windows\Fonts\tahoma.ttf",
#     r"C:\Windows\Fonts\trebuc.ttf",
#     r"C:\Windows\Fonts\cour.ttf",
# ]
#
# # 字号与描边
# FONT_SIZE_RANGE = (18, 36)
# STROKE_WIDTH_CHOICES = [1, 1, 2]
#
# # 颜色：仍然让它看起来像真 PHI
# PHI_LIKE_COLORS = [
#     (255, 255, 255),   # 白
#     (255, 255, 0),     # 黄
# ]
#
# # 也允许一部分更“偏参数字”的颜色
# AUX_COLORS = [
#     (210, 210, 210),
#     (180, 180, 180),
#     (220, 220, 160),
#     (170, 210, 210),
#     (190, 220, 190),
# ]
#
# # =========================================================
# # 模糊与困难样本参数
# # =========================================================
# TEXT_BLUR_PROB = 0.40
# TEXT_BLUR_RADIUS_RANGE = (0.4, 1.2)
#
# IMAGE_BLUR_PROB = 0.18
# IMAGE_BLUR_RADIUS_RANGE = (0.2, 0.8)
#
# # 假 PHI 数量
# FAKE_PHI_COUNT_RANGE = (6, 12)
#
# # 少量允许相互靠近/轻微重叠，提升困难度
# DECOY_OVERLAP_PROB = 0.20
#
# # meta 文件是否保存
# SAVE_META_JSON = True
#
#
# # =========================================================
# # 随机数据池
# # =========================================================
# CN_HOSPITALS = ["华东医院", "市第一人民医院", "仁和医院", "中山医学中心", "协和影像中心", "滨海医院"]
# EN_HOSPITALS = ["Central City Hospital", "Union Medical Center", "Riverside Imaging Center", "Saint Mary Hospital", "Metro General Hospital"]
#
# CN_STREETS = ["中山路", "人民路", "解放路", "和平路", "建设路", "新华路", "长江路", "延安路"]
# CN_CITIES = ["上海", "北京", "广州", "深圳", "杭州", "南京", "武汉", "成都"]
#
# EN_STREETS = ["Main St", "Broadway", "Maple Ave", "Oak St", "Pine Rd", "6th Ave", "Park Blvd"]
# EN_CITIES = ["Boston", "Seattle", "Austin", "Chicago", "San Diego", "Houston", "Phoenix"]
#
#
# # =========================================================
# # 工具函数
# # =========================================================
# def load_random_font(size: int, font_candidates):
#     available = [fp for fp in font_candidates if os.path.exists(fp)]
#     if not available:
#         return ImageFont.load_default(), "default"
#     font_path = random.choice(available)
#     return ImageFont.truetype(font_path, size), font_path
#
#
# def rand_dt(start_year=2021, end_year=2025):
#     start = datetime(start_year, 1, 1, 0, 0, 0)
#     end = datetime(end_year, 12, 31, 23, 59, 59)
#     delta = int((end - start).total_seconds())
#     return start + timedelta(seconds=random.randint(0, delta))
#
#
# def boxes_overlap(box1, box2, pad=0):
#     x1, y1, x2, y2 = box1
#     a1, b1, a2, b2 = box2
#     return not (
#         x2 + pad < a1 or
#         a2 + pad < x1 or
#         y2 + pad < b1 or
#         b2 + pad < y1
#     )
#
#
# def sample_random_xy_with_bias(text_w, text_h, img_w, img_h, corner_bias=0.5):
#     max_x = max(MARGIN, img_w - text_w - MARGIN)
#     max_y = max(MARGIN, img_h - text_h - MARGIN)
#
#     if random.random() < corner_bias:
#         corner = random.choice(["tl", "tr", "bl", "br"])
#         x_band = max(80, img_w // 5)
#         y_band = max(80, img_h // 5)
#
#         if corner == "tl":
#             x = random.randint(MARGIN, min(max_x, MARGIN + x_band))
#             y = random.randint(MARGIN, min(max_y, MARGIN + y_band))
#         elif corner == "tr":
#             x = random.randint(max(MARGIN, max_x - x_band), max_x)
#             y = random.randint(MARGIN, min(max_y, MARGIN + y_band))
#         elif corner == "bl":
#             x = random.randint(MARGIN, min(max_x, MARGIN + x_band))
#             y = random.randint(max(MARGIN, max_y - y_band), max_y)
#         else:
#             x = random.randint(max(MARGIN, max_x - x_band), max_x)
#             y = random.randint(max(MARGIN, max_y - y_band), max_y)
#     else:
#         x = random.randint(MARGIN, max_x)
#         y = random.randint(MARGIN, max_y)
#
#     return x, y
#
#
# def find_position_for_decoy(text_w, text_h, img_w, img_h, occupied_boxes):
#     allow_overlap = random.random() < DECOY_OVERLAP_PROB
#
#     for _ in range(MAX_PLACEMENT_TRIES):
#         x, y = sample_random_xy_with_bias(
#             text_w, text_h, img_w, img_h, DECOY_CORNER_BIAS
#         )
#         candidate = (x, y, x + text_w, y + text_h)
#
#         if allow_overlap:
#             return x, y, candidate
#
#         if not any(boxes_overlap(candidate, ob, pad=PLACEMENT_PAD) for ob in occupied_boxes):
#             return x, y, candidate
#
#     x, y = sample_random_xy_with_bias(
#         text_w, text_h, img_w, img_h, DECOY_CORNER_BIAS
#     )
#     candidate = (x, y, x + text_w, y + text_h)
#     return x, y, candidate
#
#
# def make_safe_stem(file_path, root_folder):
#     rel = os.path.relpath(file_path, root_folder)
#     stem = os.path.splitext(rel)[0]
#     stem = stem.replace("\\", "_").replace("/", "_")
#     return stem
#
#
# def get_unique_output_stem(out_img_dir, base_stem):
#     stem = base_stem
#     idx = 1
#     while os.path.exists(os.path.join(out_img_dir, stem + ".png")):
#         stem = f"{base_stem}_{idx}"
#         idx += 1
#     return stem
#
#
# # =========================================================
# # 假 PHI 值生成函数
# # 这些长得很像 PHI，但实际上不属于你要检的目标
# # =========================================================
# def gen_patient_position(lang):
#     if lang == "zh":
#         return random.choice(["仰卧位", "俯卧位", "头先进仰卧位", "足先进仰卧位"])
#     return random.choice(["HFS", "FFS", "Supine", "Prone"])
#
#
# def gen_body_part(lang):
#     if lang == "zh":
#         return random.choice(["胸部", "腹部", "头部", "颈部", "胸腹部", "盆腔"])
#     return random.choice(["Chest", "Abdomen", "Head", "Neck", "Chest-Abdomen", "Pelvis"])
#
#
# def gen_orientation(lang):
#     if lang == "zh":
#         return random.choice(["横断位", "冠状位", "矢状位", "斜位"])
#     return random.choice(["Axial", "Coronal", "Sagittal", "Oblique"])
#
#
# def gen_protocol_name(lang):
#     if lang == "zh":
#         return random.choice(["胸部平扫", "增强扫描", "常规重建", "薄层重建", "定位像"])
#     return random.choice(["Chest Routine", "Contrast Study", "Standard Recon", "Thin Recon", "Scout"])
#
#
# def gen_series_number(lang):
#     return str(random.randint(1, 12))
#
#
# def gen_image_number(lang):
#     return str(random.randint(1, 180))
#
#
# def gen_slice_thickness(lang):
#     return f"{random.choice([1.0, 2.5, 5.0, 7.5])}mm"
#
#
# def gen_window_width(lang):
#     return str(random.randint(200, 1200))
#
#
# def gen_window_level(lang):
#     return str(random.randint(-200, 200))
#
#
# def gen_matrix_size(lang):
#     return random.choice(["256x256", "320x320", "512x512", "1024x1024"])
#
#
# def gen_kvp(lang):
#     return str(random.choice([80, 100, 120, 140]))
#
#
# def gen_mas(lang):
#     return str(random.randint(80, 320))
#
#
# def gen_recon_kernel(lang):
#     if lang == "zh":
#         return random.choice(["标准", "软组织", "骨算法", "肺算法"])
#     return random.choice(["STANDARD", "SOFT", "BONE", "LUNG"])
#
#
# def gen_study_desc(lang):
#     if lang == "zh":
#         return random.choice(["胸部CT", "腹部CT", "头颅MR", "盆腔MR", "全身PET"])
#     return random.choice(["Chest CT", "Abdomen CT", "Brain MR", "Pelvis MR", "Whole-body PET"])
#
#
# def gen_institution_name(lang):
#     return random.choice(CN_HOSPITALS) if lang == "zh" else random.choice(EN_HOSPITALS)
#
#
# def gen_institution_address(lang):
#     if lang == "zh":
#         city = random.choice(CN_CITIES)
#         street = random.choice(CN_STREETS)
#         no = random.randint(10, 1999)
#         return f"{city}市{street}{no}号"
#     city = random.choice(EN_CITIES)
#     street = random.choice(EN_STREETS)
#     no = random.randint(10, 1999)
#     return f"{no} {street}, {city}"
#
#
# def gen_visit_time(lang):
#     dt = rand_dt(2021, 2025)
#     if lang == "zh":
#         return random.choice([
#             dt.strftime("%Y-%m-%d %H:%M"),
#             dt.strftime("%Y/%m/%d %H:%M:%S"),
#         ])
#     return random.choice([
#         dt.strftime("%Y-%m-%d %H:%M"),
#         dt.strftime("%m/%d/%Y %H:%M"),
#     ])
#
#
# def gen_study_date(lang):
#     dt = rand_dt(2021, 2025)
#     if lang == "zh":
#         return random.choice([
#             dt.strftime("%Y-%m-%d"),
#             dt.strftime("%Y/%m/%d"),
#             dt.strftime("%Y年%m月%d日")
#         ])
#     return random.choice([
#         dt.strftime("%Y-%m-%d"),
#         dt.strftime("%m/%d/%Y"),
#         dt.strftime("%Y%m%d")
#     ])
#
#
# def make_fake_phi_entry(zh_label, en_label, value_fn):
#     lang = random.choice(["zh", "en"])
#     label = zh_label if lang == "zh" else en_label
#     sep = "：" if lang == "zh" else ": "
#     value = value_fn(lang)
#     return {
#         "label": label,
#         "sep": sep,
#         "value": value,
#         "lang": lang
#     }
#
#
# def build_fake_phi_entries():
#     pool = [
#         make_fake_phi_entry("患者体位", "Patient Position", gen_patient_position),
#         make_fake_phi_entry("检查部位", "Body Part", gen_body_part),
#         make_fake_phi_entry("图像方向", "Orientation", gen_orientation),
#         make_fake_phi_entry("扫描协议", "Protocol Name", gen_protocol_name),
#         make_fake_phi_entry("序列编号", "Series Number", gen_series_number),
#         make_fake_phi_entry("图像编号", "Image Number", gen_image_number),
#         make_fake_phi_entry("层厚", "Slice Thickness", gen_slice_thickness),
#         make_fake_phi_entry("窗宽", "Window Width", gen_window_width),
#         make_fake_phi_entry("窗位", "Window Level", gen_window_level),
#         make_fake_phi_entry("矩阵大小", "Matrix", gen_matrix_size),
#         make_fake_phi_entry("管电压", "KVP", gen_kvp),
#         make_fake_phi_entry("管电流", "mAs", gen_mas),
#         make_fake_phi_entry("重建核", "Recon Kernel", gen_recon_kernel),
#         make_fake_phi_entry("检查描述", "Study Description", gen_study_desc),
#         make_fake_phi_entry("机构名称", "Institution Name", gen_institution_name),
#         make_fake_phi_entry("机构地址", "Institution Address", gen_institution_address),
#         make_fake_phi_entry("就诊时间", "Visit Time", gen_visit_time),
#         make_fake_phi_entry("检查日期", "Study Date", gen_study_date),
#     ]
#
#     k = random.randint(*FAKE_PHI_COUNT_RANGE)
#     k = min(k, len(pool))
#     selected = random.sample(pool, k)
#     random.shuffle(selected)
#     return selected
#
#
# # =========================================================
# # DICOM 预处理
# # =========================================================
# def preprocess_dicom_to_rgb(ds):
#     pixel = ds.pixel_array
#
#     if pixel.ndim == 3:
#         if pixel.shape[-1] == 3:
#             arr = pixel
#         else:
#             arr = pixel[0]
#     else:
#         arr = pixel
#
#     arr = arr.astype(np.float32)
#
#     p1, p99 = np.percentile(arr, (1, 99))
#     arr = np.clip(arr, p1, p99)
#     arr = (arr - p1) / (p99 - p1 + 1e-6)
#     arr = (arr * 255).astype(np.uint8)
#
#     if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
#         arr = 255 - arr
#
#     img = Image.fromarray(arr).resize(TARGET_SIZE, Image.Resampling.BILINEAR).convert("RGB")
#     return img
#
#
# # =========================================================
# # 绘制
# # =========================================================
# def draw_text_with_optional_blur(
#     base_img, x, y, text, font, fill,
#     stroke_width=1, stroke_fill=(0, 0, 0),
#     blur_prob=None, blur_radius_range=None
# ):
#     if blur_prob is None:
#         blur_prob = TEXT_BLUR_PROB
#     if blur_radius_range is None:
#         blur_radius_range = TEXT_BLUR_RADIUS_RANGE
#
#     dummy_draw = ImageDraw.Draw(base_img)
#     bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
#
#     text_w = max(1, bbox[2] - bbox[0])
#     text_h = max(1, bbox[3] - bbox[1])
#
#     pad = 6
#     layer = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
#     d = ImageDraw.Draw(layer)
#
#     rgba_fill = fill + (255,)
#     rgba_stroke = stroke_fill + (255,)
#
#     d.text(
#         (pad, pad),
#         text,
#         font=font,
#         fill=rgba_fill,
#         stroke_width=stroke_width,
#         stroke_fill=rgba_stroke
#     )
#
#     if random.random() < blur_prob:
#         radius = random.uniform(*blur_radius_range)
#         layer = layer.filter(ImageFilter.GaussianBlur(radius=radius))
#
#     base_img.alpha_composite(layer, (x - pad, y - pad))
#
#
# def draw_only_fake_phi(img, font_candidates):
#     """
#     只生成高仿假 PHI，不生成真实 PHI，不标注任何框
#     返回 ann_records = []，用于 YOLO 负样本
#     """
#     img = img.convert("RGBA")
#     draw = ImageDraw.Draw(img)
#     w, h = img.size
#
#     fake_entries = build_fake_phi_entries()
#     occupied_boxes = []
#     decoy_records = []
#
#     for e in fake_entries:
#         label = e["label"]
#         sep = e["sep"]
#         value = e["value"]
#         full_text = label + sep + value
#
#         font_size = random.randint(*FONT_SIZE_RANGE)
#         font, font_path = load_random_font(font_size, font_candidates)
#         stroke_width = random.choice(STROKE_WIDTH_CHOICES)
#
#         if random.random() < 0.75:
#             color = random.choice(PHI_LIKE_COLORS)
#         else:
#             color = random.choice(AUX_COLORS)
#
#         full_box = draw.textbbox((0, 0), full_text, font=font, stroke_width=stroke_width)
#         text_w = full_box[2] - full_box[0]
#         text_h = full_box[3] - full_box[1]
#
#         x, y, candidate = find_position_for_decoy(text_w, text_h, w, h, occupied_boxes)
#         occupied_boxes.append(candidate)
#
#         draw_text_with_optional_blur(
#             img, x, y, full_text,
#             font=font,
#             fill=color,
#             stroke_width=stroke_width,
#             stroke_fill=(0, 0, 0)
#         )
#
#         decoy_records.append({
#             "label_text": label,
#             "value_text": value,
#             "full_text": full_text,
#             "lang": e["lang"],
#             "font_size": font_size,
#             "font_path": font_path,
#             "bbox_xyxy": [int(x + full_box[0]), int(y + full_box[1]), int(x + full_box[2]), int(y + full_box[3])]
#         })
#
#     # 少量整图模糊
#     if random.random() < IMAGE_BLUR_PROB:
#         radius = random.uniform(*IMAGE_BLUR_RADIUS_RANGE)
#         img = img.filter(ImageFilter.GaussianBlur(radius=radius))
#
#     ann_records = []  # 关键：负样本为空标签
#     return img.convert("RGB"), ann_records, decoy_records
#
#
# # =========================================================
# # 保存
# # =========================================================
# def save_empty_yolo_label(label_path):
#     with open(label_path, "w", encoding="utf-8") as f:
#         pass
#
#
# def save_meta_json(meta_path, image_name, decoy_records):
#     obj = {
#         "image_name": image_name,
#         "negative_sample": True,
#         "class_names": ["patient_info", "time_info", "institution_info"],
#         "records": [],
#         "fake_phi_records": decoy_records
#     }
#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)
#
#
# # =========================================================
# # 单张处理：只追加负样本
# # =========================================================
# def process_dicom_negative_only(
#     dcm_path,
#     input_root,
#     out_img_dir,
#     out_label_dir,
#     out_meta_dir,
#     font_candidates,
#     variant_idx=0
# ):
#     try:
#         ds = pydicom.dcmread(dcm_path, force=True)
#         img = preprocess_dicom_to_rgb(ds)
#
#         img, ann_records, decoy_records = draw_only_fake_phi(img, font_candidates)
#
#         base_stem = make_safe_stem(dcm_path, input_root)
#         base_stem = f"{NEGATIVE_PREFIX}_{base_stem}"
#         if variant_idx > 0:
#             base_stem = f"{base_stem}_aug{variant_idx}"
#
#         final_stem = get_unique_output_stem(out_img_dir, base_stem)
#
#         img.save(os.path.join(out_img_dir, final_stem + ".png"))
#
#         # 负样本：空 txt
#         save_empty_yolo_label(os.path.join(out_label_dir, final_stem + ".txt"))
#
#         if SAVE_META_JSON:
#             save_meta_json(
#                 os.path.join(out_meta_dir, final_stem + ".json"),
#                 final_stem + ".png",
#                 decoy_records
#             )
#
#         print(f"[OK] {final_stem}")
#
#     except Exception as e:
#         print(f"[ERROR] {dcm_path} -> {e}")
#
#
# # =========================================================
# # 追加到某个 split
# # =========================================================
# def ensure_split_dirs(split_output_root):
#     out_img_dir = os.path.join(split_output_root, "images")
#     out_label_dir = os.path.join(split_output_root, "labels")
#     out_meta_dir = os.path.join(split_output_root, "meta")
#
#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_label_dir, exist_ok=True)
#     os.makedirs(out_meta_dir, exist_ok=True)
#
#     return out_img_dir, out_label_dir, out_meta_dir
#
#
# def append_negative_split(input_folder, split_output_root, font_candidates):
#     out_img_dir, out_label_dir, out_meta_dir = ensure_split_dirs(split_output_root)
#
#     dcm_files = []
#     for root, _, files in os.walk(input_folder):
#         for fn in files:
#             if fn.lower().endswith(".dcm"):
#                 dcm_files.append(os.path.join(root, fn))
#
#     print(f"\n[Append Negative Split] {split_output_root}")
#     print(f"Total DICOM files: {len(dcm_files)}")
#     print(f"NEG_PER_DCM: {NEG_PER_DCM}")
#
#     for dcm_path in dcm_files:
#         for i in range(NEG_PER_DCM):
#             process_dicom_negative_only(
#                 dcm_path=dcm_path,
#                 input_root=input_folder,
#                 out_img_dir=out_img_dir,
#                 out_label_dir=out_label_dir,
#                 out_meta_dir=out_meta_dir,
#                 font_candidates=font_candidates,
#                 variant_idx=i
#             )
#
#
# # =========================================================
# # 主函数：追加到已有数据集
# # 这里的 output_root 要指向你“已经生成好的”现有数据集根目录
# # =========================================================
# if __name__ == "__main__":
#     # 你现有的 YOLO 数据集根目录（里面已经有 train/val/test）
#     output_root = r"F:\数据集\数据集11111"
#
#     # 原始 DICOM 底图输入
#     train_input = r"F:\数据集\数据集11111\训练集"
#     val_input   = r"F:\数据集\数据集11111\验证集"
#     test_input  = r"F:\数据集\数据集11111\测试集"
#
#     # 追加负样本到现有 train
#     append_negative_split(
#         input_folder=train_input,
#         split_output_root=os.path.join(output_root, "train"),
#         font_candidates=TRAIN_FONT_CANDIDATES
#     )
#
#     # 追加负样本到现有 val
#     append_negative_split(
#         input_folder=val_input,
#         split_output_root=os.path.join(output_root, "val"),
#         font_candidates=VAL_FONT_CANDIDATES
#     )
#
#     # 追加负样本到现有 test
#     append_negative_split(
#         input_folder=test_input,
#         split_output_root=os.path.join(output_root, "test"),
#         font_candidates=TEST_FONT_CANDIDATES
#     )