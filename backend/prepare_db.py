"""Crawl the LDS website and prepare documents for embedding and storage in ChromaDB."""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from bs4 import BeautifulSoup
import logging
from pathlib import Path

LOGGER_DIR = Path(__file__).parent.parent / "logs"
CHROMA_DB_PATH_DIR = Path(__file__).parent / "chroma_db"
LLM_MODEL = "llama3"
LDS_WEBSITE = "https://ldsociety.ca/"

# set up logger that logs to logs/prepare_db.log
LOGGER = logging.getLogger(__name__)

def setup_logger(logger: logging.Logger):
    """Set up logger to log to file and console."""
    logger.setLevel(logging.DEBUG)
    if not LOGGER_DIR.exists():
        LOGGER_DIR.mkdir(parents=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOGGER_DIR / "prepare_db.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def crawl_website(website_url: str) -> list[Document]:
    """
    Crawl the given website and return a list of Documents.
    Crawling depth is set to 2. (get content from homepage and links on homepage)
    """
    LOGGER.info(f"Crawling website: {website_url}")
    loader = RecursiveUrlLoader(
        url=website_url,
        max_depth=2,  # how deep to follow links
        extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),  # clean text
    )
    documents = loader.load()
    LOGGER.debug([document.metadata["source"] for document in documents])
    LOGGER.info(f"Crawled {len(documents)} documents from {website_url}")
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better embedding."""
    LOGGER.info("Splitting documents into smaller chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    LOGGER.info(f"Split into {len(split_docs)} documents")
    return split_docs

def create_db(documents: list[Document]) -> Chroma:
    """Convert documents to embeddings and store in ChromaDB."""
    LOGGER.info("Creating embeddings from documents")
    embeddings = OllamaEmbeddings(model=LLM_MODEL)
    LOGGER.info("Storing embeddings in ChromaDB")
    chroma_db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_PATH_DIR)
    )
    LOGGER.info("ChromaDB persisted successfully")
    return chroma_db

def main():
    setup_logger(LOGGER)
    LOGGER.info("Starting to crawl LDS website")
    docs = crawl_website(LDS_WEBSITE)
    db = create_db(docs)


if __name__ == "__main__":
    # We can add args later to force recreate the DB
    if CHROMA_DB_PATH_DIR.exists():
        LOGGER.error("ChromaDB already exists. Exiting. Please remove the existing database if you want to recreate it.")
        exit()
    main()
