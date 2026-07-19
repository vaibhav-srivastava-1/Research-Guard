# ResearchGuard

ResearchGuard is a Streamlit document question-answering app. Users can create an account, upload their own documents, ask grounded questions, and revisit saved chat history.

The current sample document is about the 2008 financial crisis.

## What This Project Does

1. Lets users sign up, log in, and log out.
2. Stores each user's uploaded documents separately under `data/users`.
3. Loads `.txt`, `.md`, `.pdf`, and `.docx` files.
4. Splits each document into smaller chunks.
5. Retrieves relevant chunks using BM25 and dense retrieval.
6. Combines retrieval results with reciprocal rank fusion.
7. Reranks the best chunks.
8. Generates an answer with source citations.
9. Checks cited claims and warns when a claim is not supported.
10. Saves each user's question and answer history in a local SQLite database.

Example question:

```text
What were the main causes and consequences of the 2008 financial crisis?
```

Example output:

```text
Banks stopped lending to each other, fearing counterparty risk (source: 2008_financial_crisis_1).
```

## Project Structure

```text
.
+-- app/
|   +-- streamlit_app.py        Streamlit web app
+-- data/
|   +-- raw/                    Put source .txt documents here
|   +-- users/                  Generated per-user uploads, ignored by git
+-- src/
|   +-- agent/                  Planner, synthesizer, critic, and orchestrator
|   +-- ingestion/              Document loading and chunking
|   +-- retrieval/              BM25, dense retrieval, fusion, and reranking
|   +-- config.py               Main configuration values
|   +-- utils.py                Shared utility functions
+-- tests/                      Unit tests
+-- main.py                     Command-line entry point
+-- requirements.txt            Python dependencies
+-- .env.example                Environment variable template
+-- README.md                   Project documentation
```

## Requirements

- Python 3.12 or compatible
- A virtual environment
- Internet access for installing packages
- An API key in `.env` if you want generated LLM answers

The app can still run with local fallback logic if some heavy ML packages, such as FAISS or Torch, do not load correctly on Windows.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your key:

```text
OPENAI_API_KEY=your_api_key_here
JWT_SECRET=change_this_to_any_private_random_value
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_this_admin_password
```

Do not commit `.env`. It contains private secrets.

## Running The Web App

Start Streamlit:

```powershell
python -m streamlit run app\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

Create a new account from the sign-up tab, then log in. User accounts and chat history are stored locally in `data/researchguard.db`.

Admin access:

- If `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set, that account is created or promoted to admin when the app starts.
- If no admin account exists, the first account created from the sign-up tab becomes an admin.
- Admin users can open the Admin page to view users, document counts, query counts, recent activity, reset passwords, change roles, clear documents, clear history, and delete user accounts.

## Running From The Command Line

You can also ask a question from the terminal:

```powershell
python main.py "What caused the 2008 financial crisis?"
```

## Running Tests

```powershell
python -m pytest -q
```

Expected result:

```text
5 passed
```

## Adding Your Own Documents

1. Log in to the Streamlit app.
2. Open the Documents page.
3. Upload one or more `.txt`, `.md`, `.pdf`, or `.docx` files.
4. Open the Chat page and ask questions about your uploaded documents.

Each file becomes a document. The filename is used in citations, so a file named `climate_report.txt` may produce citations like:

```text
(source: climate_report_0)
```

## Deploying On Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from your repository.
3. Set the main file path to:

```text
app/streamlit_app.py
```

4. Add secrets in the Streamlit Cloud app settings:

```toml
OPENAI_API_KEY = "your_api_key_here"
JWT_SECRET = "replace_with_a_long_random_secret"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "replace_with_a_strong_admin_password"
```

5. Deploy the app.

Streamlit Community Cloud storage is not guaranteed to be permanent across rebuilds or restarts. For production use, replace the local SQLite database and `data/users` uploads with managed storage, such as Postgres plus S3-compatible object storage.

## How The Pipeline Works

The main flow is controlled by `src/agent/orchestrator.py`.

1. `loader.py` reads text documents.
2. `chunker.py` splits long documents into manageable chunks.
3. `planner.py` breaks a complex question into smaller sub-questions.
4. `bm25_retriever.py` finds exact keyword matches.
5. `dense_retriever.py` finds semantic matches.
6. `fusion.py` combines BM25 and dense results.
7. `reranker.py` sorts the best candidates again.
8. `synthesizer.py` writes the answer with citations.
9. `critic.py` checks whether cited chunks support the generated claims.

## Windows Compatibility Notes

Some ML packages use compiled native files. On Windows, packages such as FAISS, Torch, NumPy, Regex, or Pydantic Core can sometimes fail if the wrong wheel is installed or a file is locked during installation.

This project includes fallbacks so the app can still run:

- If FAISS does not load, dense retrieval uses NumPy scoring.
- If SentenceTransformers does not load, dense retrieval uses lightweight local token vectors.
- If the cross-encoder reranker does not load, reranking uses lexical overlap.
- If the NLI critic model does not load, verification uses a simple lexical entailment check.

These fallbacks keep the project usable for development and demos. A fully configured ML environment will use the stronger model-backed components.

## Common Problems

### `ModuleNotFoundError` for a compiled package

Reinstall the broken package inside the virtual environment. For example:

```powershell
python -m pip install --force-reinstall --no-cache-dir pydantic-core pydantic
```

### `python` is not recognized

Install Python or use the full path to your Python executable. After installing Python, restart the terminal.

### The app shows an old error

Stop Streamlit, restart it, and refresh the browser page.

## Files That Should Stay Out Of Git

These are generated or private and should not be committed:

- `.venv/`
- `.env`
- `data/users/`
- `data/researchguard.db`
- `__pycache__/`
- `.pytest_cache/`
- `*.log`

## Current Status

The project has been tested with the sample 2008 financial crisis document. The Streamlit app runs at `http://127.0.0.1:8501`, supports account creation and login/logout, accepts user uploads, retrieves document chunks, saves history, and produces cited answers.
