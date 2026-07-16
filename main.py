import sys
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.agent.orchestrator import ResearchOrchestrator
from src.config import RAW_DATA_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your complex research question here\"")
        sys.exit(1)
        
    query = sys.argv[1]
    
    logger.info("Starting ResearchGuard pipeline...")
    
    # 1. Ingestion
    docs = load_documents(RAW_DATA_DIR)
    if not docs:
        logger.error(f"No documents found in {RAW_DATA_DIR}. Please add some text files.")
        sys.exit(1)
        
    chunks = chunk_documents(docs)
    
    # 2. Orchestration
    orchestrator = ResearchOrchestrator(chunks)
    
    # 3. Execution
    logger.info(f"Processing Query: {query}")
    final_report = orchestrator.run(query)
    
    print("\n" + "="*50)
    print("FINAL VERIFIED REPORT")
    print("="*50)
    print(final_report)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
