import datetime
import html
import os
import re
import sys
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


class SafeStream:
    def __init__(self, target):
        self.target = target

    def flush(self):
        try:
            self.target.flush()
        except Exception:
            pass

    def write(self, data):
        return self.target.write(data)

    def __getattr__(self, name):
        return getattr(self.target, name)


if sys.stderr:
    sys.stderr = SafeStream(sys.stderr)

import jwt
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="ResearchGuard", page_icon=":material/security:", layout="wide")

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

load_dotenv(project_root / ".env")
load_dotenv(project_root / ".env.example")


def load_streamlit_secrets() -> None:
    try:
        for key in ("OPENAI_API_KEY", "JWT_SECRET", "GENERATOR_MODEL"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


load_streamlit_secrets()

from src.agent.orchestrator import ResearchOrchestrator
from src.auth_store import (
    add_history,
    clear_user_history,
    create_user,
    delete_user_account,
    delete_history_item,
    get_all_history,
    get_history,
    init_db,
    is_admin_user,
    list_user_summaries,
    remove_file,
    remove_user_documents,
    reset_user_password,
    safe_username,
    set_user_admin,
    verify_user,
)
from src.config import USER_DATA_DIR
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import SUPPORTED_EXTENSIONS, load_documents

SECRET_KEY = os.getenv("JWT_SECRET", "super_secret_default_key")
TOKEN_DAYS = 7


def add_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rg-bg: #f7f7fb;
            --rg-panel: #ffffff;
            --rg-panel-soft: #fff4f4;
            --rg-ink: #262a33;
            --rg-muted: #6f7480;
            --rg-line: #e3e5ea;
            --rg-primary: #e5322d;
            --rg-primary-dark: #b8201c;
            --rg-primary-soft: #fde9e8;
            --rg-accent: #344054;
            --rg-focus: #e5322d;
        }

        .stApp {
            background:
                linear-gradient(180deg, #ffffff 0, #fff4f4 16rem, #f7f7fb 34rem),
                var(--rg-bg);
            color: var(--rg-ink);
        }

        .stApp,
        .stApp p,
        .stApp span,
        .stApp div,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {
            color: var(--rg-ink);
        }

        .stCaptionContainer,
        .stCaptionContainer p,
        small,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stCaptionContainer p {
            color: var(--rg-muted) !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.96);
            border-right: 1px solid var(--rg-line);
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: var(--rg-ink);
        }

        .rg-shell {
            animation: rg-rise 520ms ease-out both;
        }

        .rg-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.1rem 0 0.4rem;
            border-bottom: 1px solid rgba(29, 36, 48, 0.1);
            margin-bottom: 1.3rem;
        }

        .rg-title {
            font-size: 2rem;
            font-weight: 760;
            letter-spacing: 0;
            margin: 0;
        }

        .rg-subtitle {
            margin: 0.2rem 0 0;
            color: var(--rg-muted);
            max-width: 54rem;
        }

        .rg-pill {
            border: 1px solid rgba(229, 50, 45, 0.24);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            background: var(--rg-primary-soft);
            color: var(--rg-primary-dark);
            white-space: nowrap;
            font-size: 0.9rem;
        }

        .rg-stat {
            padding: 0.9rem 1rem;
            border: 1px solid var(--rg-line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 14px 30px rgba(38, 42, 51, 0.06);
        }

        .rg-stat strong {
            display: block;
            font-size: 1.35rem;
            color: var(--rg-ink);
        }

        .rg-stat span {
            color: var(--rg-muted);
            font-size: 0.88rem;
        }

        .rg-history-card {
            padding: 1rem;
            border: 1px solid var(--rg-line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.98);
            margin-bottom: 0.9rem;
            animation: rg-rise 420ms ease-out both;
        }

        .rg-question {
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .rg-time {
            color: var(--rg-muted);
            font-size: 0.82rem;
        }

        .rg-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--rg-line);
            border-radius: 8px;
            background: var(--rg-panel);
        }

        .rg-admin-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 760px;
        }

        .rg-admin-table th,
        .rg-admin-table td {
            padding: 0.75rem 0.85rem;
            border-bottom: 1px solid var(--rg-line);
            text-align: left;
            color: var(--rg-ink) !important;
            font-size: 0.9rem;
            vertical-align: top;
        }

        .rg-admin-table th {
            background: var(--rg-primary-soft);
            color: var(--rg-primary-dark) !important;
            font-weight: 750;
        }

        .rg-admin-table tr:last-child td {
            border-bottom: 0;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-tertiary"] {
            border-radius: 8px;
            background: var(--rg-panel) !important;
            border: 1px solid #cfd3dc !important;
            color: var(--rg-ink) !important;
            font-weight: 650;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        .stButton > button p,
        .stDownloadButton > button p,
        .stButton > button span,
        .stDownloadButton > button span,
        [data-testid="stBaseButton-secondary"] p,
        [data-testid="stBaseButton-secondary"] span,
        [data-testid="stBaseButton-tertiary"] p,
        [data-testid="stBaseButton-tertiary"] span {
            color: var(--rg-ink) !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stBaseButton-tertiary"]:hover {
            transform: translateY(-1px);
            border-color: var(--rg-primary) !important;
            background: var(--rg-primary-soft) !important;
            color: var(--rg-primary-dark) !important;
            box-shadow: 0 10px 22px rgba(229, 50, 45, 0.14);
        }

        .stButton > button:hover p,
        .stDownloadButton > button:hover p,
        .stButton > button:hover span,
        .stDownloadButton > button:hover span,
        [data-testid="stBaseButton-secondary"]:hover p,
        [data-testid="stBaseButton-secondary"]:hover span,
        [data-testid="stBaseButton-tertiary"]:hover p,
        [data-testid="stBaseButton-tertiary"]:hover span {
            color: var(--rg-primary-dark) !important;
        }

        .stButton > button[kind="primary"],
        .stFormSubmitButton > button,
        [data-testid="stBaseButton-primary"] {
            background: var(--rg-primary) !important;
            border-color: var(--rg-primary) !important;
            color: #ffffff !important;
            font-weight: 700;
        }

        .stButton > button[kind="primary"] p,
        .stFormSubmitButton > button p,
        .stButton > button[kind="primary"] span,
        .stFormSubmitButton > button span,
        [data-testid="stBaseButton-primary"] p,
        [data-testid="stBaseButton-primary"] span {
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button:hover,
        [data-testid="stBaseButton-primary"]:hover {
            background: var(--rg-primary-dark) !important;
            border-color: var(--rg-primary-dark) !important;
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"]:hover p,
        .stFormSubmitButton > button:hover p,
        .stButton > button[kind="primary"]:hover span,
        .stFormSubmitButton > button:hover span,
        [data-testid="stBaseButton-primary"]:hover p,
        [data-testid="stBaseButton-primary"]:hover span {
            color: #ffffff !important;
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus,
        .stFormSubmitButton > button:focus,
        [data-testid^="stBaseButton"]:focus {
            box-shadow: 0 0 0 3px rgba(229, 50, 45, 0.18) !important;
            outline: none !important;
        }

        .stButton > button:disabled,
        .stDownloadButton > button:disabled,
        .stFormSubmitButton > button:disabled {
            background: #eceef3 !important;
            border-color: #d8dbe2 !important;
            color: #7c818c !important;
        }

        .stButton > button:disabled p,
        .stDownloadButton > button:disabled p,
        .stFormSubmitButton > button:disabled p,
        .stButton > button:disabled span,
        .stDownloadButton > button:disabled span,
        .stFormSubmitButton > button:disabled span {
            color: #7c818c !important;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--rg-primary);
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--rg-muted) !important;
            font-weight: 650;
        }

        .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--rg-primary-dark) !important;
        }

        .stTabs [aria-selected="true"] p {
            color: var(--rg-primary-dark) !important;
        }

        label,
        [data-testid="stTextInput"] label,
        [data-testid="stTextInput"] input,
        [data-testid="stPasswordInput"] label,
        [data-testid="stPasswordInput"] input {
            color: var(--rg-ink) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stPasswordInput"] input,
        textarea,
        [data-baseweb="input"] input {
            background: var(--rg-panel) !important;
            -webkit-text-fill-color: var(--rg-ink) !important;
            color: var(--rg-ink) !important;
            caret-color: var(--rg-primary);
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stPasswordInput"] input::placeholder,
        textarea::placeholder,
        [data-baseweb="input"] input::placeholder {
            color: var(--rg-muted) !important;
            opacity: 1;
        }

        input:focus,
        textarea:focus {
            border-color: var(--rg-focus) !important;
            box-shadow: 0 0 0 1px rgba(229, 50, 45, 0.2) !important;
        }

        div[data-testid="stFileUploader"] section {
            border-radius: 8px;
            border-color: rgba(229, 50, 45, 0.24);
            background: rgba(255, 255, 255, 0.9);
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {
            color: var(--rg-ink) !important;
        }

        [data-testid="stAlert"],
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] span {
            color: var(--rg-ink) !important;
        }

        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] span {
            color: var(--rg-ink) !important;
        }

        div[data-baseweb="radio"] label,
        div[data-baseweb="radio"] span,
        div[role="radiogroup"] label,
        div[role="radiogroup"] span {
            color: var(--rg-ink) !important;
        }

        button svg,
        [data-testid^="stBaseButton"] svg,
        [data-testid="stTextInput"] svg,
        [data-testid="stPasswordInput"] svg {
            color: var(--rg-ink) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }

        [data-testid="stPasswordInput"] button,
        [data-testid="stPasswordInput"] button:hover {
            background: transparent !important;
            border: 0 !important;
            color: var(--rg-ink) !important;
            box-shadow: none !important;
        }

        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploader"] button p,
        div[data-testid="stFileUploader"] button span {
            color: var(--rg-ink) !important;
        }

        div[data-testid="stFileUploader"] button {
            background: var(--rg-panel) !important;
            border: 1px solid #cfd3dc !important;
        }

        div[data-testid="stFileUploader"] button:hover {
            background: var(--rg-primary-soft) !important;
            border-color: var(--rg-primary) !important;
        }

        @keyframes rg-rise {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 760px) {
            .rg-topline {
                align-items: flex-start;
                flex-direction: column;
            }
            .rg-title {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_token(username: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "exp": now + datetime.timedelta(days=TOKEN_DAYS),
        "iat": now,
        "sub": username,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_token() -> str | None:
    token = st.session_state.get("token")
    if not token:
        return None
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded["sub"]
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        st.session_state["token"] = None
        return None


def current_user_dir(username: str) -> Path:
    user_dir = USER_DATA_DIR / safe_username(username)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def file_signature(user_dir: Path) -> tuple:
    signature = []
    for path in sorted(user_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            stat = path.stat()
            signature.append((path.name, stat.st_size, int(stat.st_mtime)))
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def build_orchestrator(user_dir_text: str, signature: tuple) -> ResearchOrchestrator | None:
    del signature
    docs = load_documents(Path(user_dir_text))
    if not docs:
        return None
    chunks = chunk_documents(docs)
    if not chunks:
        return None
    return ResearchOrchestrator(chunks)


def sanitize_filename(filename: str) -> str:
    clean_name = Path(filename).name
    clean_name = re.sub(r"[^A-Za-z0-9_. -]", "_", clean_name).strip()
    return clean_name or "document.txt"


def render_auth() -> None:
    st.markdown('<div class="rg-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rg-topline">
            <div>
                <h1 class="rg-title">ResearchGuard</h1>
                <p class="rg-subtitle">Sign in to upload documents, ask grounded questions, and keep your research history.</p>
            </div>
            <span class="rg-pill">Private local workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            if verify_user(username, password):
                st.session_state["token"] = generate_token(username.strip().lower())
                st.rerun()
            st.error("Invalid username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            created = st.form_submit_button("Create account", use_container_width=True)
        if created:
            ok, message = create_user(new_username, new_password)
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)


def render_header(username: str, docs_count: int, history_count: int) -> None:
    escaped_username = html.escape(username)
    st.markdown('<div class="rg-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="rg-topline">
            <div>
                <h1 class="rg-title">ResearchGuard</h1>
                <p class="rg-subtitle">Ask questions about your uploaded documents and keep every useful thread in your history.</p>
            </div>
            <span class="rg-pill">Signed in as {escaped_username}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f'<div class="rg-stat"><strong>{docs_count}</strong><span>Documents uploaded</span></div>', unsafe_allow_html=True)
    col_b.markdown(f'<div class="rg-stat"><strong>{history_count}</strong><span>Saved questions</span></div>', unsafe_allow_html=True)
    col_c.markdown('<div class="rg-stat"><strong>txt md pdf docx</strong><span>Supported uploads</span></div>', unsafe_allow_html=True)


def render_uploads(username: str, user_dir: Path) -> None:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Save uploads", type="primary"):
        saved_count = 0
        for uploaded_file in uploaded_files:
            filename = sanitize_filename(uploaded_file.name)
            target_path = user_dir / filename
            target_path.write_bytes(uploaded_file.getbuffer())
            saved_count += 1
        st.cache_resource.clear()
        st.success(f"Saved {saved_count} document(s).")
        st.rerun()

    files = sorted(
        path for path in user_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        st.info("Upload at least one document to start asking questions.")
        return

    st.caption("Your documents")
    for path in files:
        c1, c2, c3 = st.columns([6, 2, 1])
        c1.write(path.name)
        c2.caption(f"{path.stat().st_size / 1024:.1f} KB")
        if c3.button("Delete", key=f"delete_{path.name}"):
            remove_file(path)
            st.cache_resource.clear()
            st.rerun()


def render_chat(username: str, user_dir: Path) -> None:
    st.subheader("Ask Your Documents")
    signature = file_signature(user_dir)
    if not signature:
        st.info("Upload documents first. Your chat will unlock as soon as the app can index them.")
        return

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        st.warning("Set OPENAI_API_KEY in Streamlit secrets or your .env file for model-generated answers. Local fallback answers may still run.")

    with st.spinner("Indexing your documents..."):
        orchestrator = build_orchestrator(str(user_dir), signature)

    if orchestrator is None:
        st.error("No readable text was found in your uploaded documents.")
        return

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving, drafting, and verifying citations..."):
                try:
                    answer = orchestrator.run(question)
                    highlighted = answer.replace(
                        "[WARNING: UNSUPPORTED BY CITATION]",
                        "**:red[[WARNING: UNSUPPORTED BY CITATION]]**",
                    )
                    st.markdown(highlighted)
                    add_history(username, question, answer)
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")


def render_history(username: str) -> None:
    st.subheader("History")
    history = get_history(username)
    if not history:
        st.info("Your saved questions will appear here after you ask something.")
        return

    for item in history:
        escaped_question = html.escape(item["question"])
        escaped_created_at = html.escape(item["created_at"])
        st.markdown(
            f"""
            <div class="rg-history-card">
                <div class="rg-question">{escaped_question}</div>
                <div class="rg-time">{escaped_created_at}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View saved answer"):
            st.markdown(item["answer"])
            if st.button("Delete this item", key=f"history_delete_{item['id']}"):
                delete_history_item(username, item["id"])
                st.rerun()


def render_users_table(summaries: list[dict]) -> None:
    header = """
        <tr>
            <th>User</th>
            <th>Role</th>
            <th>Documents</th>
            <th>Storage</th>
            <th>Queries</th>
            <th>Created</th>
            <th>Last query</th>
        </tr>
    """
    rows = []
    for item in summaries:
        rows.append(
            f"""
            <tr>
                <td>{html.escape(str(item["username"]))}</td>
                <td>{html.escape(str(item["role"]))}</td>
                <td>{html.escape(str(item["documents"]))}</td>
                <td>{html.escape(str(item["storage_kb"]))} KB</td>
                <td>{html.escape(str(item["queries"]))}</td>
                <td>{html.escape(str(item["created_at"]))}</td>
                <td>{html.escape(str(item["last_query_at"] or "-"))}</td>
            </tr>
            """
        )
    st.markdown(
        f"""
        <div class="rg-table-wrap">
            <table class="rg-admin-table">
                <thead>{header}</thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_panel(current_username: str) -> None:
    if not is_admin_user(current_username):
        st.error("Admin access is required.")
        return

    st.subheader("Admin Panel")
    summaries = list_user_summaries(USER_DATA_DIR)
    all_history = get_all_history()

    total_docs = sum(item["documents"] for item in summaries)
    total_storage = sum(item["storage_kb"] for item in summaries)
    total_queries = sum(item["queries"] for item in summaries)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Users", len(summaries))
    col_b.metric("Documents", total_docs)
    col_c.metric("Queries", total_queries)
    col_d.metric("Storage", f"{total_storage:.1f} KB")

    users_tab, activity_tab, controls_tab = st.tabs(["Users", "Activity", "Controls"])

    with users_tab:
        if summaries:
            render_users_table(summaries)
        else:
            st.info("No users found.")

    with activity_tab:
        if not all_history:
            st.info("No saved query history yet.")
        else:
            for item in all_history:
                st.markdown(
                    f"""
                    <div class="rg-history-card">
                        <div class="rg-question">{html.escape(item["username"])} asked: {html.escape(item["question"])}</div>
                        <div class="rg-time">{html.escape(item["created_at"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("View answer"):
                    st.markdown(item["answer"])

    with controls_tab:
        usernames = [item["username"] for item in summaries]
        if not usernames:
            st.info("Create users before using admin controls.")
            return

        selected_user = st.selectbox("Select user", usernames)
        selected_summary = next(item for item in summaries if item["username"] == selected_user)

        st.markdown("#### Account")
        st.write(
            f"Role: **{selected_summary['role']}** | "
            f"Documents: **{selected_summary['documents']}** | "
            f"Queries: **{selected_summary['queries']}**"
        )

        c1, c2 = st.columns(2)
        with c1:
            make_admin = st.toggle(
                "Admin access",
                value=selected_summary["role"] == "admin",
                disabled=selected_user == current_username,
                help="You cannot remove admin access from your own signed-in account.",
            )
            if st.button("Save role", use_container_width=True, disabled=selected_user == current_username):
                set_user_admin(selected_user, make_admin)
                st.success("Role updated.")
                st.rerun()

        with c2:
            with st.form("admin_password_reset"):
                new_password = st.text_input("New password", type="password")
                submitted = st.form_submit_button("Reset password", use_container_width=True)
            if submitted:
                ok, message = reset_user_password(selected_user, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)

        st.markdown("#### Data")
        d1, d2, d3 = st.columns(3)
        if d1.button("Clear documents", use_container_width=True):
            remove_user_documents(selected_user, USER_DATA_DIR)
            st.cache_resource.clear()
            st.success("Documents cleared.")
            st.rerun()
        if d2.button("Clear history", use_container_width=True):
            clear_user_history(selected_user)
            st.success("History cleared.")
            st.rerun()
        delete_disabled = selected_user == current_username
        if d3.button("Delete user", use_container_width=True, disabled=delete_disabled):
            remove_user_documents(selected_user, USER_DATA_DIR)
            delete_user_account(selected_user)
            st.cache_resource.clear()
            st.success("User deleted.")
            st.rerun()

        if delete_disabled:
            st.caption("You cannot delete the admin account you are currently using.")


def render_app(username: str) -> None:
    user_dir = current_user_dir(username)
    docs_count = len(file_signature(user_dir))
    history_count = len(get_history(username))
    admin = is_admin_user(username)

    with st.sidebar:
        st.markdown("### ResearchGuard")
        pages = ["Chat", "Documents", "History"]
        if admin:
            pages.append("Admin")
        page = st.radio("Navigation", pages, label_visibility="collapsed")
        st.divider()
        st.caption(f"Logged in as {username}")
        if admin:
            st.caption("Admin access enabled")
        if st.button("Log out", use_container_width=True):
            st.session_state["token"] = None
            st.rerun()

    render_header(username, docs_count, history_count)

    if page == "Chat":
        render_chat(username, user_dir)
    elif page == "Documents":
        render_uploads(username, user_dir)
    elif page == "History":
        render_history(username)
    else:
        render_admin_panel(username)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_db()
    add_global_styles()
    username = decode_token()
    if username is None:
        render_auth()
        st.stop()
    render_app(username)


if __name__ == "__main__":
    main()
