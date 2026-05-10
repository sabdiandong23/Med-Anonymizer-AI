import sqlite3
from datetime import datetime
import uuid
import os
import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DB_PATH = "security.db"

# 32字节密钥：演示阶段先写死
# 实际应用中建议放到环境变量或独立配置文件中
AES_KEY = b"0123456789abcdef0123456789abcdef"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL,
        input_path TEXT,
        output_path TEXT,
        role TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        key_id TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS encrypted_fields (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,
        field_name TEXT NOT NULL,
        original_value_type TEXT,
        nonce TEXT NOT NULL,
        ciphertext TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES cases(case_id)
    )
    """)

    conn.commit()
    conn.close()


def encrypt_text(plain_text: str):
    """
    用 AES-GCM 加密字符串
    返回: nonce_b64, ciphertext_b64
    """
    aesgcm = AESGCM(AES_KEY)

    # GCM 推荐 12 字节 nonce
    nonce = os.urandom(12)

    # aad=None，演示阶段先不加附加认证数据
    ciphertext = aesgcm.encrypt(nonce, plain_text.encode("utf-8"), None)

    nonce_b64 = base64.b64encode(nonce).decode("utf-8")
    ciphertext_b64 = base64.b64encode(ciphertext).decode("utf-8")
    return nonce_b64, ciphertext_b64


def decrypt_text(nonce_b64: str, ciphertext_b64: str):
    """
    反向解密
    """
    aesgcm = AESGCM(AES_KEY)

    nonce = base64.b64decode(nonce_b64.encode("utf-8"))
    ciphertext = base64.b64decode(ciphertext_b64.encode("utf-8"))

    plain_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    return plain_bytes.decode("utf-8")


def save_case_and_fields(file_name, input_path, output_path, role, field_dict):
    """
    保存一次病例处理记录，以及该病例对应的原始敏感字段加密结果
    """
    conn = get_conn()
    cur = conn.cursor()

    case_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat(timespec="seconds")

    # 1) 插入 cases 表
    cur.execute("""
        INSERT INTO cases (
            case_id, file_name, input_path, output_path,
            role, algorithm, key_id, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        file_name,
        input_path,
        output_path,
        role,
        "AES-GCM",
        "demo_key_01",
        created_at
    ))

    # 2) 插入 encrypted_fields 表
    for field_name, value in field_dict.items():
        if value is None:
            continue

        value_str = str(value).strip()
        if not value_str:
            continue

        nonce_b64, ciphertext_b64 = encrypt_text(value_str)

        cur.execute("""
            INSERT INTO encrypted_fields (
                case_id, field_name, original_value_type, nonce, ciphertext
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            case_id,
            field_name,
            type(value).__name__,
            nonce_b64,
            ciphertext_b64
        ))

    conn.commit()
    conn.close()
    return case_id


def fetch_case(case_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()

    conn.close()
    return row


def fetch_encrypted_fields(case_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, case_id, field_name, original_value_type, nonce, ciphertext
        FROM encrypted_fields
        WHERE case_id = ?
        ORDER BY id
    """, (case_id,))
    rows = cur.fetchall()

    conn.close()
    return rows


def decrypt_case_fields(case_id):
    """
    查询并解密某个 case_id 对应的全部字段
    返回列表，每项是 dict
    """
    rows = fetch_encrypted_fields(case_id)
    result = []

    for row in rows:
        field_name = row[2]
        original_value_type = row[3]
        nonce_b64 = row[4]
        ciphertext_b64 = row[5]

        try:
            plain_text = decrypt_text(nonce_b64, ciphertext_b64)
        except Exception as e:
            plain_text = f"[解密失败] {e}"

        result.append({
            "field_name": field_name,
            "original_value_type": original_value_type,
            "nonce": nonce_b64,
            "ciphertext": ciphertext_b64,
            "plain_text": plain_text,
        })

    return result


def show_all_cases():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM cases ORDER BY created_at DESC")
    case_rows = cur.fetchall()

    print("\n=== cases 表数据 ===")
    for row in case_rows:
        print(row)

    cur.execute("SELECT * FROM encrypted_fields ORDER BY id DESC")
    field_rows = cur.fetchall()

    print("\n=== encrypted_fields 表数据 ===")
    for row in field_rows:
        print(row)

    conn.close()


def show_case_by_id(case_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    case_row = cur.fetchone()

    print("\n=== 指定 case 信息 ===")
    print(case_row)

    cur.execute("""
        SELECT id, case_id, field_name, original_value_type, nonce, ciphertext
        FROM encrypted_fields
        WHERE case_id = ?
        ORDER BY id
    """, (case_id,))
    field_rows = cur.fetchall()

    print("\n=== 指定 case 的字段信息 ===")
    for row in field_rows:
        print(row)

    conn.close()


def clear_all_data():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM encrypted_fields")
    cur.execute("DELETE FROM cases")

    conn.commit()
    conn.close()
    print("已清空 cases 和 encrypted_fields 表数据。")


if __name__ == "__main__":
    init_db()
    print("数据库初始化完成。")