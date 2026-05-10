import io
import os
import zipfile
import tempfile
import sqlite3

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ===== 兼容补丁：适配 streamlit-drawable-canvas 的旧版 image_to_url 调用 =====
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig


def _image_to_url_compat(
    image,
    width,
    clamp=False,
    channels="RGB",
    output_format="auto",
    image_id=None,
):
    layout_width = width if width is not None else "content"
    return _new_image_to_url(
        image,
        layout_config=LayoutConfig(width=layout_width),
        clamp=clamp,
        channels=channels,
        output_format=output_format,
        image_id=image_id,
    )


st_image.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas

from doc_head import DICOMAnonymizer
from picture import (
    preprocess_dicom,
    detect_regions,
    draw_detection_preview,
    build_manual_detections_from_canvas,
    save_dicom_with_detections,
    process_dicom,
)
from security_store import decrypt_text

MODEL_PATH = "D:/Users/Desktop/作业/毕业设计/module/baseline/best.pt"
DB_PATH = "security.db"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def init_state():
    defaults = {
        "single_job": None,
        "single_result": None,
        "single_manual_detections": [],
        "manual_canvas_version": 0,
        "batch_result": None,
        "security_query_rows": [],
        "security_plain_rows": [],
        "security_last_case_id": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_single_state():
    st.session_state.single_job = None
    st.session_state.single_result = None
    st.session_state.single_manual_detections = []
    st.session_state.manual_canvas_version = 0
    st.session_state.security_query_rows = []
    st.session_state.security_plain_rows = []
    st.session_state.security_last_case_id = ""


def role_to_policy(role):
    return {
        "doctor": ["patient_info"],
        "researcher": ["patient_info", "time_info", "institution_info"],
    }[role]


def build_strategy_text(role):
    if role == "doctor":
        return """
**当前策略：医生**
- 图像：仅模糊 `patient_info`
- 文件头：仅处理患者相关字段
- 适合临床查看场景
"""
    return """
**当前策略：科研人员**
- 图像：模糊 `patient_info / time_info / institution_info`
- 文件头：处理患者、时间、机构等敏感字段
- 适合科研数据共享场景
"""


def human_cls_name(name):
    mapping = {
        "patient_info": "患者信息",
        "time_info": "时间信息",
        "institution_info": "机构信息",
    }
    return mapping.get(name, name)


def fetch_encrypted_fields(case_id: str):
    if not case_id:
        return []

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, field_name, original_value_type, nonce, ciphertext
        FROM encrypted_fields
        WHERE case_id = ?
        ORDER BY id
    """, (case_id,))
    rows = cur.fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "field_name": row[1],
            "original_value_type": row[2],
            "nonce": row[3],
            "ciphertext": row[4],
        })
    return result


def decrypt_case_fields(case_id: str):
    rows = fetch_encrypted_fields(case_id)
    plain_rows = []

    for row in rows:
        try:
            plain_text = decrypt_text(row["nonce"], row["ciphertext"])
        except Exception as e:
            plain_text = f"[解密失败] {e}"

        plain_rows.append({
            "field_name": row["field_name"],
            "original_value": plain_text,
        })

    return rows, plain_rows


def mask_ciphertext(text: str, head=16):
    if not text:
        return ""
    if len(text) <= head:
        return text
    return text[:head] + "..."


st.set_page_config(
    page_title="DICOM De-identification System",
    page_icon="🩺",
    layout="wide",
)

init_state()
model = load_model()
anonymizer = DICOMAnonymizer()

st.title("DICOM De-identification System")
st.caption("文件头脱敏 + 像素级 PHI 检测与模糊 + 手动补漏 + 安全恢复验证")


# =============================
# Sidebar 参数区 + Form
# =============================
with st.sidebar:
    st.header("参数区")

    mode = st.radio(
        "选择处理模式",
        ["单个文件", "批量文件"],
        horizontal=True
    )

    with st.form("control_form"):
        role_label = st.radio(
            "选择身份",
            ["医生", "科研人员"],
            horizontal=True
        )

        conf_thres = st.slider(
            "检测置信度阈值",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05
        )

        blur_kernel = st.slider(
            "模糊强度（核大小）",
            min_value=11,
            max_value=99,
            value=51,
            step=2
        )

        save_folder = st.text_input("输出保存文件夹（可选）", "")

        submitted = st.form_submit_button(
            "开始处理",
            width="stretch",
            type="primary"
        )

    ROLE_MAP = {
        "医生": "doctor",
        "科研人员": "researcher"
    }
    role = ROLE_MAP[role_label]
    blur_classes = role_to_policy(role)

    st.markdown("---")
    st.markdown(build_strategy_text(role))
    st.info("黄色框 = 手动补漏框")


# =============================
# 单个文件模式：自动检测 + 手动补漏
# =============================
if mode == "单个文件":
    st.subheader("单个文件处理")
    uploaded_file = st.file_uploader("上传 DICOM 文件", type=["dcm"], key="single_uploader")

    if submitted:
        if uploaded_file is None:
            st.warning("请先上传一个 DICOM 文件。")
        else:
            with st.spinner("正在执行自动检测..."):
                reset_single_state()

                tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
                tmp_input.write(uploaded_file.read())
                tmp_input.close()

                tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
                tmp_output.close()

                success, changes = anonymizer.anonymize_dicom(
                    tmp_input.name,
                    tmp_input.name,
                    role=role
                )

                current_case_id = getattr(anonymizer, "last_case_id", "")

                import pydicom
                ds = pydicom.dcmread(tmp_input.name)
                before_image = preprocess_dicom(ds.pixel_array)

                auto_detections = detect_regions(
                    before_image,
                    model,
                    blur_classes=blur_classes,
                    conf=conf_thres
                )

                st.session_state.single_job = {
                    "file_name": uploaded_file.name,
                    "tmp_input_path": tmp_input.name,
                    "tmp_output_path": tmp_output.name,
                    "before_image": before_image,
                    "auto_detections": auto_detections,
                    "changes": changes,
                    "save_folder": save_folder,
                    "blur_kernel": blur_kernel,
                    "case_id": current_case_id,
                }
                st.session_state.security_last_case_id = current_case_id

    single_job = st.session_state.single_job

    if single_job:
        before_image = single_job["before_image"]
        auto_detections = single_job["auto_detections"]
        manual_detections = st.session_state.single_manual_detections
        merged_detections = auto_detections + manual_detections

        auto_preview = draw_detection_preview(before_image, auto_detections)
        merged_preview = draw_detection_preview(before_image, merged_detections)

        c1, c2, c3 = st.columns(3)
        c1.metric("自动检测框", len(auto_detections))
        c2.metric("手动补漏框", len(manual_detections))
        c3.metric("合并后总框数", len(merged_detections))

        st.markdown("### 检测预览")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("处理前")
            st.image(before_image, clamp=True, width="stretch")

        with col2:
            st.subheader("自动检测框")
            st.image(auto_preview, clamp=True, width="stretch")

        with col3:
            st.subheader("自动 + 手动 合并预览")
            st.image(merged_preview, clamp=True, width="stretch")

        st.markdown("---")
        st.markdown("### 手动补漏")
        st.caption("如果模型漏掉了敏感文字，请先选择类别，再在下图上画矩形框，然后点击“添加手动框”。")

        manual_class = st.selectbox(
            "手动补漏类别",
            options=blur_classes,
            format_func=human_cls_name,
            key="manual_class_select"
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.15)",
            stroke_width=2,
            stroke_color="#FFFF00",
            background_image=Image.fromarray(before_image),
            update_streamlit=True,
            height=before_image.shape[0],
            width=before_image.shape[1],
            drawing_mode="rect",
            display_toolbar=True,
            key=f"manual_canvas_{st.session_state.manual_canvas_version}",
        )

        btn1, btn2, btn3, btn4 = st.columns(4)

        with btn1:
            if st.button("添加手动框", width="stretch"):
                new_manual = build_manual_detections_from_canvas(
                    canvas_result.json_data if canvas_result else None,
                    manual_class=manual_class,
                    image_shape=before_image.shape
                )

                if not new_manual:
                    st.warning("你还没有画框。")
                else:
                    st.session_state.single_manual_detections.extend(new_manual)
                    st.session_state.single_result = None
                    st.session_state.manual_canvas_version += 1
                    st.rerun()

        with btn2:
            if st.button("删除最后一个框", width="stretch"):
                if st.session_state.single_manual_detections:
                    st.session_state.single_manual_detections.pop()
                    st.session_state.single_result = None
                    st.session_state.manual_canvas_version += 1
                    st.rerun()
                else:
                    st.warning("当前没有可删除的手动框。")

        with btn3:
            if st.button("清空手动框", width="stretch"):
                st.session_state.single_manual_detections = []
                st.session_state.single_result = None
                st.session_state.manual_canvas_version += 1
                st.rerun()

        with btn4:
            if st.button("生成最终脱敏结果", type="primary", width="stretch"):
                with st.spinner("正在生成最终脱敏结果..."):
                    final_result = save_dicom_with_detections(
                        input_path=single_job["tmp_input_path"],
                        output_path=single_job["tmp_output_path"],
                        detections=merged_detections,
                        blur_kernel=single_job["blur_kernel"]
                    )

                    with open(single_job["tmp_output_path"], "rb") as f:
                        output_bytes = f.read()

                    saved_path = ""
                    if single_job["save_folder"]:
                        os.makedirs(single_job["save_folder"], exist_ok=True)
                        saved_path = os.path.join(
                            single_job["save_folder"],
                            f"processed_{single_job['file_name']}"
                        )
                        with open(saved_path, "wb") as f:
                            f.write(output_bytes)

                    st.session_state.single_result = {
                        "file_name": single_job["file_name"],
                        "before_image": final_result["before_image"],
                        "preview_image": final_result["preview_image"],
                        "after_image": final_result["after_image"],
                        "class_counts": final_result["class_counts"],
                        "num_boxes": final_result["num_boxes"],
                        "changes": single_job["changes"],
                        "output_bytes": output_bytes,
                        "saved_path": saved_path,
                        "case_id": single_job.get("case_id", ""),
                    }
                    st.rerun()

        if manual_detections:
            st.markdown("### 已添加的手动框")
            rows = []
            for i, det in enumerate(manual_detections, start=1):
                x1, y1, x2, y2 = det["box"]
                rows.append({
                    "序号": i,
                    "类别": det["class_name"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")

    single_result = st.session_state.single_result
    if single_result:
        st.markdown("---")
        st.markdown("### 最终脱敏结果")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总框数", single_result["num_boxes"])
        c2.metric("patient_info", single_result["class_counts"].get("patient_info", 0))
        c3.metric("time_info", single_result["class_counts"].get("time_info", 0))
        c4.metric("institution_info", single_result["class_counts"].get("institution_info", 0))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("处理前")
            st.image(single_result["before_image"], clamp=True, width="stretch")

        with col2:
            st.subheader("最终框预览")
            st.image(single_result["preview_image"], clamp=True, width="stretch")

        with col3:
            st.subheader("处理后")
            st.image(single_result["after_image"], clamp=True, width="stretch")

        st.markdown("### 文件头变化")
        df = pd.DataFrame(single_result["changes"], columns=["Tag", "Before", "After"])
        st.dataframe(df, width="stretch")

        case_id = single_result.get("case_id", "")
        if case_id:
            st.success(f"安全数据库记录已生成，case_id: {case_id}")

        if single_result["saved_path"]:
            st.success(f"已保存到: {single_result['saved_path']}")

        st.download_button(
            "下载处理后文件",
            data=single_result["output_bytes"],
            file_name=f"processed_{single_result['file_name']}",
            mime="application/octet-stream"
        )

        # =============================
        # 安全恢复验证（新增）
        # =============================
        st.markdown("---")
        with st.expander("安全恢复验证（仅授权）", expanded=False):
            st.caption("该模块用于演示：系统会将原始敏感字段以 AES-GCM 密文形式存入数据库，并可在授权条件下恢复。")

            default_case_id = case_id if case_id else st.session_state.security_last_case_id
            query_case_id = st.text_input(
                "输入 case_id",
                value=default_case_id,
                key="security_case_id_input"
            )

            b1, b2, b3 = st.columns(3)

            with b1:
                if st.button("查询加密字段", width="stretch"):
                    rows = fetch_encrypted_fields(query_case_id)
                    st.session_state.security_query_rows = rows
                    st.session_state.security_plain_rows = []
                    if not rows:
                        st.warning("未查询到该 case_id 对应的加密字段。")

            with b2:
                if st.button("授权解密", type="primary", width="stretch"):
                    rows, plain_rows = decrypt_case_fields(query_case_id)
                    st.session_state.security_query_rows = rows
                    st.session_state.security_plain_rows = plain_rows
                    if not rows:
                        st.warning("未查询到该 case_id 对应的加密字段。")

            with b3:
                if st.button("清空恢复结果", width="stretch"):
                    st.session_state.security_query_rows = []
                    st.session_state.security_plain_rows = []
                    st.rerun()

            query_rows = st.session_state.security_query_rows
            plain_rows = st.session_state.security_plain_rows

            if query_rows:
                st.markdown("#### 数据库中的加密字段")
                enc_df = pd.DataFrame([
                    {
                        "字段名": r["field_name"],
                        "类型": r["original_value_type"],
                        "nonce": r["nonce"],
                        "ciphertext(预览)": mask_ciphertext(r["ciphertext"]),
                    }
                    for r in query_rows
                ])
                st.dataframe(enc_df, width="stretch")

            if plain_rows:
                st.markdown("#### 授权恢复结果")
                plain_df = pd.DataFrame([
                    {
                        "字段名": r["field_name"],
                        "恢复出的原始值": r["original_value"],
                    }
                    for r in plain_rows
                ])
                st.dataframe(plain_df, width="stretch")


# =============================
# 批量文件模式：保持自动处理
# =============================
if mode == "批量文件":
    st.subheader("批量文件处理")
    uploaded_files = st.file_uploader(
        "上传多个 DICOM 文件",
        type=["dcm"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if submitted:
        if not uploaded_files:
            st.warning("请先上传多个 DICOM 文件。")
        else:
            progress = st.progress(0)
            status_text = st.empty()

            output_dir = tempfile.mkdtemp()
            result_info = {}
            summary_rows = []

            for idx, file in enumerate(uploaded_files, start=1):
                status_text.info(f"正在处理: {file.name} ({idx}/{len(uploaded_files)})")

                try:
                    input_path = os.path.join(output_dir, file.name)
                    with open(input_path, "wb") as f:
                        f.write(file.read())

                    success, changes = anonymizer.anonymize_dicom(
                        input_path,
                        input_path,
                        role=role
                    )

                    output_path = os.path.join(output_dir, "processed_" + file.name)

                    process_result = process_dicom(
                        input_path,
                        output_path,
                        model,
                        blur_classes=blur_classes,
                        conf=conf_thres,
                        blur_kernel=blur_kernel
                    )

                    with open(output_path, "rb") as f:
                        output_bytes = f.read()

                    saved_path = ""
                    if save_folder:
                        os.makedirs(save_folder, exist_ok=True)
                        saved_path = os.path.join(save_folder, "processed_" + file.name)
                        with open(saved_path, "wb") as f:
                            f.write(output_bytes)

                    result_info[file.name] = {
                        "before_image": process_result["before_image"],
                        "preview_image": process_result["preview_image"],
                        "after_image": process_result["after_image"],
                        "class_counts": process_result["class_counts"],
                        "num_boxes": process_result["num_boxes"],
                        "changes": changes,
                        "output_bytes": output_bytes,
                        "saved_path": saved_path,
                    }

                    summary_rows.append({
                        "文件名": file.name,
                        "状态": "成功",
                        "检测框数": process_result["num_boxes"],
                        "patient_info": process_result["class_counts"].get("patient_info", 0),
                        "time_info": process_result["class_counts"].get("time_info", 0),
                        "institution_info": process_result["class_counts"].get("institution_info", 0),
                        "Header修改数": len(changes),
                    })

                except Exception as e:
                    summary_rows.append({
                        "文件名": file.name,
                        "状态": f"失败: {e}",
                        "检测框数": 0,
                        "patient_info": 0,
                        "time_info": 0,
                        "institution_info": 0,
                        "Header修改数": 0,
                    })

                progress.progress(idx / len(uploaded_files))

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for name, info in result_info.items():
                    zipf.writestr("processed_" + name, info["output_bytes"])
            zip_bytes = zip_buffer.getvalue()

            st.session_state.batch_result = {
                "summary_df": pd.DataFrame(summary_rows),
                "result_info": result_info,
                "zip_bytes": zip_bytes,
            }

            status_text.success("批量处理完成。")

    batch_result = st.session_state.batch_result
    if batch_result:
        st.markdown("### 批量结果总览表")
        st.dataframe(batch_result["summary_df"], width="stretch")

        st.download_button(
            "下载全部结果（ZIP）",
            data=batch_result["zip_bytes"],
            file_name="result.zip",
            mime="application/zip"
        )

        success_files = list(batch_result["result_info"].keys())
        if success_files:
            selected = st.selectbox("选择要查看的文件", success_files)

            if selected:
                info = batch_result["result_info"][selected]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("检测框总数", info["num_boxes"])
                c2.metric("patient_info", info["class_counts"].get("patient_info", 0))
                c3.metric("time_info", info["class_counts"].get("time_info", 0))
                c4.metric("institution_info", info["class_counts"].get("institution_info", 0))

                st.markdown("### 选中文件预览")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("处理前")
                    st.image(info["before_image"], clamp=True, width="stretch")

                with col2:
                    st.subheader("检测框预览")
                    st.image(info["preview_image"], clamp=True, width="stretch")

                with col3:
                    st.subheader("处理后")
                    st.image(info["after_image"], clamp=True, width="stretch")

                st.markdown("### 文件头变化")
                df = pd.DataFrame(info["changes"], columns=["Tag", "Before", "After"])
                st.dataframe(df, width="stretch")

                if info["saved_path"]:
                    st.success(f"已保存到: {info['saved_path']}")

                st.download_button(
                    "下载当前选中文件",
                    data=info["output_bytes"],
                    file_name=f"processed_{selected}",
                    mime="application/octet-stream"
                )