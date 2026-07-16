import streamlit as st
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
from pathlib import Path

# SafeStream proxy to handle Streamlit's incompatible stderr.flush() on Windows
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
import datetime
from dotenv import load_dotenv

# Add project root to sys.path so we can import src
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Load environment variables robustly using absolute paths
load_dotenv(project_root / ".env")
load_dotenv(project_root / ".env.example")

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.agent.orchestrator import ResearchOrchestrator
from src.config import RAW_DATA_DIR

st.set_page_config(page_title="ResearchGuard", page_icon="🛡️", layout="wide")

# Basic JWT Auth
SECRET_KEY = os.getenv("JWT_SECRET", "super_secret_default_key")

def generate_token(username):
    payload = {
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1),
        "iat": datetime.datetime.utcnow(),
        "sub": username
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

if "token" not in st.session_state:
    st.session_state["token"] = None

if not st.session_state["token"]:
    st.title("🛡️ ResearchGuard - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Hardcoded demo credentials
        if username == "admin" and password == "admin":
            st.session_state["token"] = generate_token(username)
            st.rerun()
        else:
            st.error("Invalid credentials (use admin/admin)")
    st.stop()

try:
    decoded = jwt.decode(st.session_state["token"], SECRET_KEY, algorithms=["HS256"])
    st.sidebar.success(f"Logged in as {decoded['sub']}")
    if st.sidebar.button("Logout"):
        st.session_state["token"] = None
        st.rerun()
except jwt.ExpiredSignatureError:
    st.session_state["token"] = None
    st.warning("Session expired. Please log in again.")
    st.stop()
except jwt.InvalidTokenError:
    st.session_state["token"] = None
    st.error("Invalid token. Please log in again.")
    st.stop()

st.title("🛡️ ResearchGuard")
st.markdown("Agentic Research & Fact-Verification Assistant")

@st.cache_resource
def init_orchestrator():
    docs = load_documents(RAW_DATA_DIR)
    if not docs:
        st.error(f"No documents found in {RAW_DATA_DIR}")
        return None
    chunks = chunk_documents(docs)
    return ResearchOrchestrator(chunks)

orchestrator = init_orchestrator()

if orchestrator is None:
    st.stop()

query = st.text_input("Ask a research question about your documents:")

if st.button("Research & Verify"):
    if not query:
        st.warning("Please enter a query.")
    elif not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        st.error("Please set a valid OPENAI_API_KEY in the .env file.")
    else:
        with st.spinner("Decomposing question, retrieving context, drafting, and checking entailment..."):
            try:
                final_report = orchestrator.run(query)
                st.subheader("Final Verified Report")
                
                # Highlight [WARNING: UNSUPPORTED] for visual effect
                highlighted_report = final_report.replace(
                    "[WARNING: UNSUPPORTED BY CITATION]", 
                    "**:red[[WARNING: UNSUPPORTED BY CITATION]]**"
                )
                
                st.markdown(highlighted_report)
            except Exception as e:
                st.error(f"An error occurred: {e}")
