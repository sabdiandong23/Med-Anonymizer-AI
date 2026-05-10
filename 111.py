import numpy as np
import pydicom
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pydicom.uid import ExplicitVRLittleEndian


# ===== 1) 这里写死你要显示的内容 =====
FIXED_LINES = [
    "Patient Name: MARTIN^CHAD",
    "patient_id: 339833062",
    #"Visit Date: 2013-07-13",
    #"Study Time: 15:50:09",
    #"Institution Name: Kim, Jones and Frazier Medical Center",
    #"Operator Name: REESE^STEPHEN",
    #"Referring Physicia: HAAS^WILLIAM",
    #"Date of Birth: 1976-06-16",
    #"Sex : male",
]

# ===== 2) 左上角固定位置参数 =====
LEFT = 18
TOP = 18
PAD_X = 16
PAD_Y = 12
LINE_GAP = 6
FONT_SIZE = 26          # 固定文字大小，不自适应
BOX_ALPHA = 170         # 背景透明度，越大越黑


def first_value(v):
    if isinstance(v, (list, tuple)):
        return v[0]
    try:
        return v[0]
    except Exception:
        return v


def get_font(font_size=26, font_path=None):
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(str(font_path), font_size)

    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            pass

    return ImageFont.load_default()


def dicom_to_uint8(ds):
    arr = ds.pixel_array.astype(np.float32)

    # 如果是 RGB，就直接返回
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    # 多帧灰度
    if arr.ndim == 3 and arr.shape[-1] != 3:
        frames = []
        for i in range(arr.shape[0]):
            frame = arr[i]

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            frame = frame * slope + intercept

            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)

            if wc is not None and ww is not None:
                wc = float(first_value(wc))
                ww = float(first_value(ww))
                low = wc - ww / 2
                high = wc + ww / 2
            else:
                low, high = np.percentile(frame, (1, 99))

            frame = np.clip(frame, low, high)
            frame = (frame - low) / (high - low + 1e-8) * 255.0

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                frame = 255 - frame

            frames.append(frame.astype(np.uint8))
        return np.stack(frames, axis=0)

    # 单帧灰度
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        wc = float(first_value(wc))
        ww = float(first_value(ww))
        low = wc - ww / 2
        high = wc + ww / 2
    else:
        low, high = np.percentile(arr, (1, 99))

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8) * 255.0

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr

    return arr.astype(np.uint8)


def resize_image_to_1024(img_array):
    """
    统一缩放到 1024x1024
    - 灰度: H x W
    - RGB: H x W x 3
    - 多帧灰度: F x H x W
    """
    target_size = (1024, 1024)

    # 单帧灰度
    if img_array.ndim == 2:
        img = Image.fromarray(img_array)
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # RGB
    elif img_array.ndim == 3 and img_array.shape[-1] == 3:
        img = Image.fromarray(img_array)
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # 多帧灰度
    elif img_array.ndim == 3:
        out_frames = []
        for i in range(img_array.shape[0]):
            img = Image.fromarray(img_array[i])
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            out_frames.append(np.array(img, dtype=np.uint8))
        return np.stack(out_frames, axis=0)

    else:
        raise ValueError(f"不支持的图像维度: {img_array.shape}")


def draw_fixed_text_on_image(img_array, font):
    """
    img_array:
      - 灰度: H x W
      - RGB:  H x W x 3
    只绘制文字，不绘制黑色背景框
    """
    if img_array.ndim == 2:
        base = Image.fromarray(img_array).convert("RGBA")
        out_mode = "L"
    else:
        base = Image.fromarray(img_array).convert("RGBA")
        out_mode = "RGB"

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bboxes = [draw.textbbox((0, 0), line, font=font) for line in FIXED_LINES]
    heights = [b[3] - b[1] for b in bboxes]

    y = TOP + PAD_Y
    for i, line in enumerate(FIXED_LINES):
        draw.text(
            (LEFT + PAD_X, y),
            line,
            fill=(230, 230, 230, 255),   # 文字颜色
            font=font
        )
        y += heights[i] + LINE_GAP

    out = Image.alpha_composite(base, overlay)

    if out_mode == "L":
        return np.array(out.convert("L"), dtype=np.uint8)
    return np.array(out.convert("RGB"), dtype=np.uint8)


def convert_back_to_dicom(ds, arr8):
    """
    直接把处理后的图写回 DICOM 像素。
    统一转成 8-bit MONOCHROME2 或 RGB。
    """
    if arr8.ndim == 2:
        ds.Rows, ds.Columns = arr8.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        if hasattr(ds, "PlanarConfiguration"):
            del ds.PlanarConfiguration
        ds.PixelData = arr8.tobytes()

    elif arr8.ndim == 3 and arr8.shape[-1] == 3:
        ds.Rows, ds.Columns = arr8.shape[:2]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = arr8.tobytes()

    elif arr8.ndim == 3:
        # 多帧灰度: F x H x W
        ds.NumberOfFrames = arr8.shape[0]
        ds.Rows, ds.Columns = arr8.shape[1], arr8.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        if hasattr(ds, "PlanarConfiguration"):
            del ds.PlanarConfiguration
        ds.PixelData = arr8.tobytes()

    for tag_name in [
        "RescaleIntercept",
        "RescaleSlope",
        "WindowCenter",
        "WindowWidth",
        "VOILUTFunction",
    ]:
        if hasattr(ds, tag_name):
            delattr(ds, tag_name)

    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def annotate_one_dicom(in_path, out_path, font):
    ds = pydicom.dcmread(str(in_path), force=True)

    try:
        ds.decompress()
    except Exception:
        pass

    if "PixelData" not in ds:
        raise ValueError("该文件没有 PixelData")

    # 1) DICOM 转可显示图像
    arr8 = dicom_to_uint8(ds)

    # 2) 统一图片大小到 1024x1024
    arr8 = resize_image_to_1024(arr8)

    # 3) 左上角添加固定文字（文字大小不变）
    if arr8.ndim == 3 and arr8.shape[-1] != 3:
        out_frames = []
        for i in range(arr8.shape[0]):
            out_frames.append(draw_fixed_text_on_image(arr8[i], font))
        arr8 = np.stack(out_frames, axis=0)
    else:
        arr8 = draw_fixed_text_on_image(arr8, font)

    # 4) 写回 DICOM
    ds = convert_back_to_dicom(ds, arr8)
    ds.save_as(str(out_path), write_like_original=False)


def process_dicom_folder(input_folder, output_folder=None, overwrite=False, recursive=True, font_path=None):
    """
    overwrite=False: 保存到 output_folder
    overwrite=True : 直接覆盖原文件
    """
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")

    if not overwrite:
        if output_folder is None:
            raise ValueError("overwrite=False 时，必须提供 output_folder")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    font = get_font(FONT_SIZE, font_path)

    files = [p for p in (input_folder.rglob("*") if recursive else input_folder.glob("*")) if p.is_file()]
    total = len(files)
    ok = 0
    skip = 0

    for idx, f in enumerate(files, 1):
        try:
            if overwrite:
                out_path = f
            else:
                rel = f.relative_to(input_folder)
                out_path = output_folder / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)

            annotate_one_dicom(f, out_path, font)
            ok += 1
            print(f"[{idx}/{total}] 成功: {f}")

        except Exception as e:
            skip += 1
            print(f"[{idx}/{total}] 跳过: {f} | 原因: {e}")

    print(f"\n处理完成：成功 {ok}，跳过 {skip}")


if __name__ == "__main__":
    process_dicom_folder(
        input_folder=r"F:\数据集\real训练集\1002.000000-NA-53238",                      # 原始DICOM文件夹
        output_folder=r"F:\数据集\out",                # 输出文件夹
        overwrite=False,                                         # True=直接覆盖原文件
        recursive=True,                                          # True=递归处理子文件夹
        font_path=None
    )