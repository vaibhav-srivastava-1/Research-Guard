# ResearchGuard

ResearchGuard is a simple document question-answering app. It reads text files from `data/raw`, retrieves the most relevant chunks, generates an answer, and checks whether each cited sentence is supported by the source text.

The current sample document is about the 2008 financial crisis.

## What This Project Does

1. Loads `.txt` files from `data/raw`.
2. Splits each document into smaller chunks.
3. Retrieves relevant chunks using BM25 and dense retrieval.
4. Combines retrieval results with reciprocal rank fusion.
5. Reranks the best chunks.
6. Generates an answer with source citations.
7. Checks cited claims and warns when a claim is not supported.

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

Demo login:

```text
Username: admin
Password: admin
```

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

1. Put one or more `.txt` files inside `data/raw`.
2. Restart the Streamlit app.
3. Ask questions about the new documents.

Each file becomes a document. The filename is used in citations, so a file named `climate_report.txt` may produce citations like:

```text
(source: climate_report_0)
```

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
- `__pycache__/`
- `.pytest_cache/`
- `*.log`

## Current Status

The project has been tested with the sample 2008 financial crisis document. The Streamlit app runs at `http://127.0.0.1:8501`, accepts the demo login, retrieves document chunks, and produces cited answers.
