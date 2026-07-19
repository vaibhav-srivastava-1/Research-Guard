from pathlib import Path

import src.auth_store as auth_store
import src.config as config


def use_temp_db(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "researchguard.db"
    monkeypatch.setattr(config, "APP_DB_PATH", db_path)
    monkeypatch.setattr(auth_store, "APP_DB_PATH", db_path)
    return db_path


def test_admin_account_and_controls(monkeypatch, tmp_path):
    use_temp_db(monkeypatch, tmp_path)
    user_data_dir = tmp_path / "users"
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")

    auth_store.init_db()

    assert auth_store.verify_user("admin", "secret123")
    assert auth_store.is_admin_user("admin")

    ok, message = auth_store.create_user("alice", "password1")
    assert ok, message
    assert not auth_store.is_admin_user("alice")

    alice_dir = user_data_dir / auth_store.safe_username("alice")
    alice_dir.mkdir(parents=True)
    (alice_dir / "notes.txt").write_text("sample", encoding="utf-8")
    auth_store.add_history("alice", "Question?", "Answer.")

    summaries = auth_store.list_user_summaries(user_data_dir)
    alice = next(item for item in summaries if item["username"] == "alice")
    assert alice["documents"] == 1
    assert alice["queries"] == 1

    auth_store.set_user_admin("alice", True)
    assert auth_store.is_admin_user("alice")

    ok, message = auth_store.reset_user_password("alice", "newpass1")
    assert ok, message
    assert auth_store.verify_user("alice", "newpass1")

    auth_store.clear_user_history("alice")
    assert auth_store.get_history("alice") == []

    auth_store.remove_user_documents("alice", user_data_dir)
    assert not alice_dir.exists()

    auth_store.delete_user_account("alice")
    summaries = auth_store.list_user_summaries(user_data_dir)
    assert {item["username"] for item in summaries} == {"admin"}


def test_sqlite_connections_close_after_admin_operations(monkeypatch, tmp_path):
    db_path = use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")

    auth_store.init_db()
    auth_store.create_user("alice", "password1")
    auth_store.add_history("alice", "Question?", "Answer.")
    auth_store.list_user_summaries(tmp_path / "users")
    auth_store.delete_user_account("alice")

    db_path.unlink()
    assert not db_path.exists()
