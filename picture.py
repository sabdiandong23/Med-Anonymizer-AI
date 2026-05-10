#
#
# from collections import Counter
#
# import cv2
# import numpy as np
# import pydicom
# from PIL import Image
#
# TARGET_SIZE = (1024, 1024)
#
#
# def preprocess_dicom(pixel_array):
#     """DICOM 像素预处理：归一化到 uint8 并缩放到固定尺寸"""
#     img = pixel_array.astype(np.float32)
#
#     p1, p99 = np.percentile(img, (1, 99))
#     img = np.clip(img, p1, p99)
#     img = (img - p1) / (p99 - p1 + 1e-6)
#     img = (img * 255).astype(np.uint8)
#
#     img = Image.fromarray(img)
#     img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
#
#     return np.array(img)
#
#
# def get_class_name(model, cls_id):
#     """从 YOLO 模型中获取类别名，兼容 dict / list"""
#     names = model.names
#     cls_id = int(cls_id)
#
#     if isinstance(names, dict):
#         return names.get(cls_id, str(cls_id))
#     if isinstance(names, list):
#         if 0 <= cls_id < len(names):
#             return names[cls_id]
#     return str(cls_id)
#
#
# def ensure_rgb(img):
#     """灰度图转 RGB，便于绘制检测框和送入 YOLO"""
#     if len(img.shape) == 2:
#         return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     return img.copy()
#
#
# def safe_blur_roi(roi, blur_kernel=51):
#     """安全模糊，避免核大小超过 ROI 尺寸"""
#     h, w = roi.shape[:2]
#     if h < 3 or w < 3:
#         return roi
#
#     k = min(
#         blur_kernel,
#         h if h % 2 == 1 else h - 1,
#         w if w % 2 == 1 else w - 1
#     )
#
#     if k < 3:
#         return roi
#     if k % 2 == 0:
#         k -= 1
#     if k < 3:
#         return roi
#
#     return cv2.GaussianBlur(roi, (k, k), 0)
#
#
# def detect_regions(img, model, blur_classes=None, conf=0.25):
#     """
#     检测图像中的敏感区域
#
#     返回格式：
#     [
#         {
#             "box": [x1, y1, x2, y2],
#             "cls_id": 0,
#             "class_name": "patient_info",
#             "conf": 0.91
#         },
#         ...
#     ]
#     """
#     if blur_classes is not None:
#         blur_classes = set(blur_classes)
#
#     img_3 = ensure_rgb(img)
#     results = model(img_3, conf=conf, verbose=False)[0]
#
#     detections = []
#     if results.boxes is None or len(results.boxes) == 0:
#         return detections
#
#     boxes_xyxy = results.boxes.xyxy.cpu().numpy()
#     boxes_cls = results.boxes.cls.cpu().numpy()
#     boxes_conf = results.boxes.conf.cpu().numpy()
#
#     h, w = img.shape[:2]
#
#     for box, cls_id, score in zip(boxes_xyxy, boxes_cls, boxes_conf):
#         class_name = get_class_name(model, cls_id)
#
#         if blur_classes is not None and class_name not in blur_classes:
#             continue
#
#         x1, y1, x2, y2 = map(int, box)
#
#         # 边界保护
#         x1 = max(0, min(x1, w - 1))
#         y1 = max(0, min(y1, h - 1))
#         x2 = max(0, min(x2, w))
#         y2 = max(0, min(y2, h))
#
#         if x2 <= x1 or y2 <= y1:
#             continue
#
#         detections.append(
#             {
#                 "box": [x1, y1, x2, y2],
#                 "cls_id": int(cls_id),
#                 "class_name": class_name,
#                 "conf": float(score),
#             }
#         )
#
#     return detections
#
#
# def draw_detection_preview(img, detections):
#     """绘制检测框预览图"""
#     preview = ensure_rgb(img)
#
#     color_map = {
#         "patient_info": (255, 0, 0),       # 红
#         "time_info": (0, 255, 0),          # 绿
#         "institution_info": (0, 0, 255),   # 蓝
#     }
#
#     for det in detections:
#         x1, y1, x2, y2 = det["box"]
#         class_name = det["class_name"]
#         score = det["conf"]
#
#         color = color_map.get(class_name, (255, 255, 0))
#         cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
#
#         text = f"{class_name} {score:.2f}"
#         text_y = max(20, y1 - 8)
#         cv2.putText(
#             preview,
#             text,
#             (x1, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             color,
#             2,
#             cv2.LINE_AA
#         )
#
#     return preview
#
#
# def apply_mask(img, detections, blur_kernel=51):
#     """按检测框对敏感区域做模糊"""
#     masked = img.copy()
#
#     for det in detections:
#         x1, y1, x2, y2 = det["box"]
#         roi = masked[y1:y2, x1:x2]
#         if roi.size == 0:
#             continue
#
#         blur = safe_blur_roi(roi, blur_kernel=blur_kernel)
#         masked[y1:y2, x1:x2] = blur
#
#     return masked
#
#
# def process_dicom(input_path, output_path, model, blur_classes=None, conf=0.25, blur_kernel=51):
#     """
#     处理单个 DICOM 文件
#
#     参数：
#         input_path: 输入 DICOM 路径
#         output_path: 输出 DICOM 路径
#         model: YOLO 模型
#         blur_classes: 要模糊的类别列表
#         conf: 检测置信度阈值
#         blur_kernel: 模糊核大小
#
#     返回：
#         {
#             "before_image": np.ndarray,
#             "preview_image": np.ndarray,
#             "after_image": np.ndarray,
#             "class_counts": dict,
#             "num_boxes": int
#         }
#     """
#     ds = pydicom.dcmread(input_path)
#     pixel = ds.pixel_array
#
#     img = preprocess_dicom(pixel)
#     detections = detect_regions(img, model, blur_classes=blur_classes, conf=conf)
#     preview_img = draw_detection_preview(img, detections)
#     masked_img = apply_mask(img, detections, blur_kernel=blur_kernel)
#
#     # 写回 DICOM
#     ds.Rows = TARGET_SIZE[1]
#     ds.Columns = TARGET_SIZE[0]
#     ds.BitsAllocated = 8
#     ds.BitsStored = 8
#     ds.HighBit = 7
#     ds.PixelRepresentation = 0
#     ds.PhotometricInterpretation = "MONOCHROME2"
#     ds.PixelData = masked_img.tobytes()
#     ds.save_as(output_path)
#
#     class_counts = dict(Counter([d["class_name"] for d in detections]))
#
#     return {
#         "before_image": img,
#         "preview_image": preview_img,
#         "after_image": masked_img,
#         "class_counts": class_counts,
#         "num_boxes": len(detections),
#     }


from collections import Counter

import cv2
import numpy as np
import pydicom
from PIL import Image

TARGET_SIZE = (1024, 1024)


def preprocess_dicom(pixel_array):
    """DICOM 像素预处理：归一化到 uint8 并缩放到固定尺寸"""
    img = pixel_array.astype(np.float32)

    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-6)
    img = (img * 255).astype(np.uint8)

    img = Image.fromarray(img)
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

    return np.array(img)


def get_class_name(model, cls_id):
    """从 YOLO 模型中获取类别名，兼容 dict / list"""
    names = model.names
    cls_id = int(cls_id)

    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, list):
        if 0 <= cls_id < len(names):
            return names[cls_id]
    return str(cls_id)


def ensure_rgb(img):
    """灰度图转 RGB，便于绘制检测框和送入 YOLO"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img.copy()


def safe_blur_roi(roi, blur_kernel=51):
    """安全模糊，避免核大小超过 ROI 尺寸"""
    h, w = roi.shape[:2]
    if h < 3 or w < 3:
        return roi

    k = min(
        blur_kernel,
        h if h % 2 == 1 else h - 1,
        w if w % 2 == 1 else w - 1
    )

    if k < 3:
        return roi
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return roi

    return cv2.GaussianBlur(roi, (k, k), 0)


def detect_regions(img, model, blur_classes=None, conf=0.25):
    """
    检测图像中的敏感区域

    返回格式：
    [
        {
            "box": [x1, y1, x2, y2],
            "cls_id": 0,
            "class_name": "patient_info",
            "conf": 0.91,
            "source": "auto"
        },
        ...
    ]
    """
    if blur_classes is not None:
        blur_classes = set(blur_classes)

    img_3 = ensure_rgb(img)
    results = model(img_3, conf=conf, verbose=False)[0]

    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    boxes_cls = results.boxes.cls.cpu().numpy()
    boxes_conf = results.boxes.conf.cpu().numpy()

    h, w = img.shape[:2]

    for box, cls_id, score in zip(boxes_xyxy, boxes_cls, boxes_conf):
        class_name = get_class_name(model, cls_id)

        if blur_classes is not None and class_name not in blur_classes:
            continue

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "cls_id": int(cls_id),
                "class_name": class_name,
                "conf": float(score),
                "source": "auto",
            }
        )

    return detections


def build_manual_detections_from_canvas(json_data, manual_class, image_shape):
    """
    把 streamlit-drawable-canvas 的矩形对象转成检测框列表
    """
    if not json_data or "objects" not in json_data:
        return []

    h, w = image_shape[:2]
    detections = []

    for obj in json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue

        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        width = float(obj.get("width", 0))
        height = float(obj.get("height", 0))
        scale_x = float(obj.get("scaleX", 1))
        scale_y = float(obj.get("scaleY", 1))

        x1 = max(0, min(int(round(left)), w - 1))
        y1 = max(0, min(int(round(top)), h - 1))
        x2 = max(0, min(int(round(left + width * scale_x)), w))
        y2 = max(0, min(int(round(top + height * scale_y)), h))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "cls_id": -1,
                "class_name": manual_class,
                "conf": 1.0,
                "source": "manual",
            }
        )

    return detections


def summarize_detections(detections):
    return dict(Counter([d["class_name"] for d in detections]))


def draw_detection_preview(img, detections):
    """
    自动框：
      patient_info 红
      time_info 绿
      institution_info 蓝

    手动框：
      黄色，并带 [M] 标记
    """
    preview = ensure_rgb(img)

    auto_color_map = {
        "patient_info": (255, 0, 0),
        "time_info": (0, 255, 0),
        "institution_info": (0, 0, 255),
    }

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_name = det["class_name"]
        score = det["conf"]
        source = det.get("source", "auto")

        if source == "manual":
            color = (255, 255, 0)  # 黄
            text = f"[M] {class_name}"
        else:
            color = auto_color_map.get(class_name, (255, 255, 0))
            text = f"{class_name} {score:.2f}"

        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

        text_y = max(20, y1 - 8)
        cv2.putText(
            preview,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    return preview


def apply_mask(img, detections, blur_kernel=51):
    """按检测框对敏感区域做模糊"""
    masked = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        roi = masked[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        blur = safe_blur_roi(roi, blur_kernel=blur_kernel)
        masked[y1:y2, x1:x2] = blur

    return masked


def save_dicom_with_detections(input_path, output_path, detections, blur_kernel=51):
    """
    用指定检测框列表生成最终脱敏 DICOM
    """
    ds = pydicom.dcmread(input_path)
    pixel = ds.pixel_array

    img = preprocess_dicom(pixel)
    preview_img = draw_detection_preview(img, detections)
    masked_img = apply_mask(img, detections, blur_kernel=blur_kernel)

    ds.Rows = TARGET_SIZE[1]
    ds.Columns = TARGET_SIZE[0]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = masked_img.tobytes()
    ds.save_as(output_path)

    class_counts = summarize_detections(detections)

    return {
        "before_image": img,
        "preview_image": preview_img,
        "after_image": masked_img,
        "class_counts": class_counts,
        "num_boxes": len(detections),
    }


def process_dicom(input_path, output_path, model, blur_classes=None, conf=0.25, blur_kernel=51):
    """
    自动检测 + 自动脱敏
    """
    ds = pydicom.dcmread(input_path)
    pixel = ds.pixel_array

    img = preprocess_dicom(pixel)
    detections = detect_regions(img, model, blur_classes=blur_classes, conf=conf)

    return save_dicom_with_detections(
        input_path=input_path,
        output_path=output_path,
        detections=detections,
        blur_kernel=blur_kernel
    )
