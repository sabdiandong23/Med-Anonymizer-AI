import os
import glob
import hashlib
import random
from pathlib import Path
from datetime import datetime, timedelta

import pydicom

from security_store import init_db, save_case_and_fields

# 确保数据库和表存在
init_db()


class DICOMAnonymizer:
    def __init__(self):
        # 定义需要脱敏的标签及其策略，并加上 category
        self.sensitive_tags = {
            # =====================
            # 患者信息
            # =====================
            (0x0010, 0x0010): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'patient'},  # PatientName
            (0x0010, 0x0020): {'strategy': 'hash',         'pattern': r'.*', 'category': 'patient'},  # PatientID
            (0x0010, 0x0030): {'strategy': 'offset_date',  'pattern': r'.*', 'category': 'patient'},  # PatientBirthDate
            (0x0010, 0x0040): {'strategy': 'delete',       'pattern': r'.*', 'category': 'patient'},  # PatientSex

            # =====================
            # 时间信息
            # =====================
            (0x0008, 0x0020): {'strategy': 'offset_date', 'pattern': r'.*', 'category': 'time'},  # StudyDate
            (0x0008, 0x0021): {'strategy': 'offset_date', 'pattern': r'.*', 'category': 'time'},  # SeriesDate
            (0x0008, 0x0022): {'strategy': 'offset_date', 'pattern': r'.*', 'category': 'time'},  # AcquisitionDate
            (0x0008, 0x0030): {'strategy': 'offset_time', 'pattern': r'.*', 'category': 'time'},  # StudyTime
            (0x0008, 0x0031): {'strategy': 'offset_time', 'pattern': r'.*', 'category': 'time'},  # SeriesTime

            # =====================
            # 机构 / 医生信息
            # =====================
            (0x0008, 0x0080): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'institution'},  # InstitutionName
            (0x0008, 0x0081): {'strategy': 'delete',       'pattern': r'.*', 'category': 'institution'},  # InstitutionAddress
            (0x0008, 0x0090): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'institution'},  # ReferringPhysicianName
            (0x0008, 0x1050): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'institution'},  # PerformingPhysicianName
            (0x0008, 0x1070): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'institution'},  # OperatorsName

            # =====================
            # 设备 / 描述信息
            # =====================
            (0x0008, 0x1010): {'strategy': 'pseudonymize', 'pattern': r'.*', 'category': 'device'},  # StationName
            (0x0008, 0x1030): {'strategy': 'keep',         'pattern': r'.*', 'category': 'device'},  # StudyDescription
            (0x0008, 0x103E): {'strategy': 'keep',         'pattern': r'.*', 'category': 'device'},  # SeriesDescription
        }

        self.patient_categories = {'patient'}
        self.researcher_categories = {'patient', 'time', 'institution', 'device'}

        # 伪名化字典（确保同一患者的多次检查使用相同伪名）
        self.pseudonym_map = {}
        self.date_offset = random.randint(-365, 365)

        # 可选：保存最近一次写库生成的 case_id
        self.last_case_id = None

    def pseudonymize(self, original_value, tag_name):
        """伪名化处理：生成虚构但一致的值"""
        if original_value in self.pseudonym_map:
            return self.pseudonym_map[original_value]

        if 'Name' in tag_name:
            first_names = ['John', 'Michael', 'David', 'James', 'Robert']
            last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Anderson']
            new_value = f'Patient_{random.choice(first_names)}{random.choice(last_names)}'
        elif 'Institution' in tag_name:
            institutions = [
                'General Hospital',
                'City Medical Center',
                'Regional Health Institute',
                'University Hospital',
                'Central Clinic'
            ]
            new_value = f'Research_{random.choice(institutions)}'
        else:
            new_value = f'Anon_{hash(original_value) % 10000:04d}'

        self.pseudonym_map[original_value] = new_value
        return new_value

    def hash_value(self, original_value):
        return hashlib.sha256(original_value.encode()).hexdigest()[:16]

    def offset_date(self, original_date):
        try:
            original_dt = datetime.strptime(original_date, '%Y%m%d')
            new_dt = original_dt + timedelta(days=self.date_offset)
            return new_dt.strftime('%Y%m%d')
        except Exception:
            return '19000101'

    def offset_time(self, original_time):
        try:
            time_part = original_time.split('.')[0] if '.' in original_time else original_time
            if len(time_part) == 6:
                original_dt = datetime.strptime(time_part, '%H%M%S')
                offset_minutes = random.randint(-120, 120)
                new_dt = original_dt + timedelta(minutes=offset_minutes)
                return new_dt.strftime('%H%M%S')
        except Exception:
            pass
        return '000000'

    def offset_age(self, original_age):
        try:
            age_value = int(original_age[:-1])
            age_unit = original_age[-1]
            offset = random.randint(-5, 5)
            new_age = max(0, min(99, age_value + offset))
            return f"{new_age:03d}{age_unit}"
        except Exception:
            return "030Y"

    def get_allowed_categories(self, role):
        if role == "doctor":
            return self.patient_categories
        return self.researcher_categories

    def safe_get(self, dataset, tag_name, default=""):
        """安全读取 DICOM 字段"""
        if hasattr(dataset, tag_name):
            value = getattr(dataset, tag_name)
            if value is None:
                return default
            return str(value).strip()
        return default

    def extract_sensitive_fields(self, dataset, role):
        """
        提取要写入安全数据库的原始敏感字段
        注意：这里提取的是“原始值”，必须发生在脱敏之前
        """
        fields = {
            "PatientName": self.safe_get(dataset, "PatientName"),
            "PatientID": self.safe_get(dataset, "PatientID"),
            "PatientBirthDate": self.safe_get(dataset, "PatientBirthDate"),
            "PatientSex": self.safe_get(dataset, "PatientSex"),
        }

        if role == "researcher":
            fields.update({
                "StudyDate": self.safe_get(dataset, "StudyDate"),
                "SeriesDate": self.safe_get(dataset, "SeriesDate"),
                "AcquisitionDate": self.safe_get(dataset, "AcquisitionDate"),
                "StudyTime": self.safe_get(dataset, "StudyTime"),
                "SeriesTime": self.safe_get(dataset, "SeriesTime"),
                "InstitutionName": self.safe_get(dataset, "InstitutionName"),
                "InstitutionAddress": self.safe_get(dataset, "InstitutionAddress"),
                "ReferringPhysicianName": self.safe_get(dataset, "ReferringPhysicianName"),
                "PerformingPhysicianName": self.safe_get(dataset, "PerformingPhysicianName"),
                "OperatorsName": self.safe_get(dataset, "OperatorsName"),
                "StationName": self.safe_get(dataset, "StationName"),
            })

        return fields

    def anonymize_dicom(self, input_path, output_path, role="researcher"):
        """
        role:
            doctor      -> 只处理患者信息
            researcher  -> 处理全部敏感信息

        返回:
            success, changes_log
        """
        changes_log = []

        try:
            dataset = pydicom.dcmread(input_path)
            allowed_categories = self.get_allowed_categories(role)

            print(f"正在处理: {input_path}")
            print(f"当前角色: {role}")
            print("=" * 50)

            # =============================
            # 1. 先提取原始敏感字段并写入安全数据库
            # =============================
            original_fields = self.extract_sensitive_fields(dataset, role)

            case_id = save_case_and_fields(
                file_name=Path(input_path).name,
                input_path=str(input_path),
                output_path=str(output_path),
                role=role,
                field_dict=original_fields
            )
            self.last_case_id = case_id
            print(f"安全数据库写入完成，case_id = {case_id}")
            print("=" * 50)

            for elem in dataset.iterall():
                tag = (elem.tag.group, elem.tag.element)

                if tag not in self.sensitive_tags:
                    continue

                strategy_info = self.sensitive_tags[tag]
                category = strategy_info.get("category", "")

                # 按角色过滤
                if category not in allowed_categories:
                    continue

                original_value = str(elem.value) if elem.value is not None else ""
                tag_name = elem.name if elem.name else f"({elem.tag.group:04x},{elem.tag.element:04x})"

                print(f"处理标签: {tag_name}")
                print(f"原始值: {original_value}")

                new_value = self.apply_strategy(original_value, tag_name, strategy_info)

                print(f"处理后: {new_value}")
                print("-" * 30)

                changes_log.append((tag_name, original_value, new_value))

                if strategy_info['strategy'] != 'delete':
                    elem.value = new_value
                else:
                    elem.value = ""

            # 添加脱敏标识
            comment_prefix = f"De-identified file (role: {role})"
            if 'ImageComments' in dataset and dataset.ImageComments:
                dataset.ImageComments = comment_prefix + " - " + str(dataset.ImageComments)
            else:
                dataset.ImageComments = comment_prefix

            dataset.save_as(output_path)
            print(f"脱敏完成！输出文件: {output_path}")
            return True, changes_log

        except Exception as e:
            print(f"处理文件时出错: {e}")
            return False, changes_log

    def apply_strategy(self, value, tag_name, strategy_info):
        strategy = strategy_info['strategy']

        if strategy == 'keep':
            return value
        elif strategy == 'delete':
            return ""
        elif strategy == 'pseudonymize':
            return self.pseudonymize(value, tag_name)
        elif strategy == 'hash':
            return self.hash_value(value)
        elif strategy == 'offset_date':
            return self.offset_date(value)
        elif strategy == 'offset_time':
            return self.offset_time(value)
        elif strategy == 'offset_age':
            return self.offset_age(value)
        else:
            return "ANONYMIZED"


# 使用示例
def main():
    anonymizer = DICOMAnonymizer()

    input_file = r"F:\数据集\测试集\Pseudo-PHI-DICOM-Data\Pseudo-PHI-001\06-26-2003-NA-XR CHEST AP PORTABLE-96544\1001.000000-NA-42825\1-1.dcm"
    output_file = r"F:\数据集\测试集\Pseudo-PHI-DICOM-Data\Pseudo-PHI-001\06-26-2003-NA-XR CHEST AP PORTABLE-96544\1001.000000-NA-42825\anonymized.dcm"

    success, changes = anonymizer.anonymize_dicom(input_file, output_file, role="researcher")

    if success:
        print("脱敏处理成功完成！")
        print(f"最近一次写库 case_id: {anonymizer.last_case_id}")
        print("\n验证脱敏结果:")
        print("=" * 50)

        original_ds = pydicom.dcmread(input_file)
        anonymized_ds = pydicom.dcmread(output_file)

        tags_to_compare = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0020),  # PatientID
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0008, 0x0020),  # StudyDate
        ]

        for tag in tags_to_compare:
            original_elem = original_ds.get(tag)
            anonymized_elem = anonymized_ds.get(tag)

            original_val = str(original_elem.value) if original_elem else "N/A"
            anonymized_val = str(anonymized_elem.value) if anonymized_elem else "N/A"

            tag_name = original_elem.name if original_elem else f"Tag {tag}"
            print(f"{tag_name}: {original_val} -> {anonymized_val}")


def batch_anonymize(input_folder, output_folder, role="researcher"):
    anonymizer = DICOMAnonymizer()
    os.makedirs(output_folder, exist_ok=True)

    dicom_files = glob.glob(os.path.join(input_folder, "*.dcm")) + \
                  glob.glob(os.path.join(input_folder, "*.DCM"))

    for input_path in dicom_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)

        print(f"\n处理文件: {filename}")
        anonymizer.anonymize_dicom(input_path, output_path, role=role)


if __name__ == "__main__":
    batch_anonymize(
        r"F:\数据集\real训练集\1.000000-PET AC-80118",
        r"F:\数据集\output_folder",
        role="researcher"
    )