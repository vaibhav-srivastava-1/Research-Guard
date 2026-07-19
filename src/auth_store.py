import hashlib
import hmac
import os
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import APP_DB_PATH
from src.ingestion.loader import SUPPORTED_EXTENSIONS


def _connect() -> sqlite3.Connection:
    APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db_connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "is_admin" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )
        _ensure_env_admin(conn)


def _ensure_env_admin(conn: sqlite3.Connection) -> None:
    admin_username = os.getenv("ADMIN_USERNAME", "").strip().lower()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if not admin_username or not admin_password:
        return

    existing = conn.execute(
        "SELECT username FROM users WHERE username = ?",
        (admin_username,),
    ).fetchone()

    salt = secrets.token_hex(16)
    password_hash = _hash_password(admin_password, salt)

    if existing:
        conn.execute(
            """
            UPDATE users
            SET is_admin = 1, password_hash = ?, salt = ?
            WHERE username = ?
            """,
            (password_hash, salt, admin_username),
        )
        return

    conn.execute(
        """
        INSERT INTO users (username, password_hash, salt, created_at, is_admin)
        VALUES (?, ?, ?, ?, 1)
        """,
        (
            admin_username,
            password_hash,
            salt,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _admin_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
    return row is not None


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    )
    return digest.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    try:
        with _db_connection() as conn:
            is_admin = 0 if _admin_exists(conn) else 1
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, created_at, is_admin)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    salt,
                    datetime.now(timezone.utc).isoformat(),
                    is_admin,
                ),
            )
        if is_admin:
            return True, "Account created. You are the first user, so admin access was enabled."
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "That username already exists."


def verify_user(username: str, password: str) -> bool:
    username = username.strip().lower()
    with _db_connection() as conn:
        user = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not user:
        return False

    attempted_hash = _hash_password(password, user["salt"])
    return hmac.compare_digest(attempted_hash, user["password_hash"])


def is_admin_user(username: str) -> bool:
    username = username.strip().lower()
    with _db_connection() as conn:
        user = conn.execute(
            "SELECT is_admin FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return bool(user and user["is_admin"])


def set_user_admin(username: str, is_admin: bool) -> None:
    with _db_connection() as conn:
        conn.execute(
            "UPDATE users SET is_admin = ? WHERE username = ?",
            (1 if is_admin else 0, username.strip().lower()),
        )


def reset_user_password(username: str, new_password: str) -> tuple[bool, str]:
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    with _db_connection() as conn:
        conn.execute(
            """
            UPDATE users
            SET password_hash = ?, salt = ?
            WHERE username = ?
            """,
            (password_hash, salt, username.strip().lower()),
        )
    return True, "Password updated."


def add_history(username: str, question: str, answer: str) -> None:
    with _db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_history (username, question, answer, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                question,
                answer,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_history(username: str, limit: int = 100) -> list[dict]:
    with _db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, created_at
            FROM chat_history
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (username, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_history(limit: int = 250) -> list[dict]:
    with _db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, username, question, answer, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_history_item(username: str, history_id: int) -> None:
    with _db_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE username = ? AND id = ?",
            (username, history_id),
        )


def clear_user_history(username: str) -> None:
    with _db_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE username = ?",
            (username.strip().lower(),),
        )


def delete_user_account(username: str) -> None:
    username = username.strip().lower()
    with _db_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))


def list_user_summaries(user_data_dir: Path) -> list[dict]:
    with _db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                users.username,
                users.created_at,
                users.is_admin,
                COUNT(chat_history.id) AS query_count,
                MAX(chat_history.created_at) AS last_query_at
            FROM users
            LEFT JOIN chat_history ON chat_history.username = users.username
            GROUP BY users.username, users.created_at, users.is_admin
            ORDER BY users.created_at DESC
            """
        ).fetchall()

    summaries = []
    for row in rows:
        username = row["username"]
        upload_dir = user_data_dir / safe_username(username)
        documents = [
            path for path in upload_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ] if upload_dir.exists() else []
        summaries.append({
            "username": username,
            "role": "admin" if row["is_admin"] else "user",
            "documents": len(documents),
            "storage_kb": round(sum(path.stat().st_size for path in documents) / 1024, 1),
            "queries": row["query_count"],
            "created_at": row["created_at"],
            "last_query_at": row["last_query_at"] or "",
        })
    return summaries


def safe_username(username: str) -> str:
    allowed = []
    for char in username.strip().lower():
        allowed.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(allowed) or "user"


def remove_file(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def remove_user_documents(username: str, user_data_dir: Path) -> None:
    user_dir = user_data_dir / safe_username(username)
    if user_dir.exists():
        shutil.rmtree(user_dir)
