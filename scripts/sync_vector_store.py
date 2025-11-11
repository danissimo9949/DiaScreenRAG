#!/usr/bin/env python
"""
Utility script that rebuilds processed chunks JSON and the persistent Chroma
vector database using PDF files from DiaScreenRAG/data/raw.

Usage:
    python scripts/sync_vector_store.py

You can optionally override default paths:
    python scripts/sync_vector_store.py --raw-dir /path/to/raw --processed-json /tmp/processed.json
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from medical_assistant.core.data_processing import process_pdfs_to_chunks
from medical_assistant.core.embeddings import create_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_PROCESSED_JSON = BASE_DIR / "data" / "processed" / "processed_pdfs.json"
DEFAULT_VECTOR_DB = BASE_DIR / "data" / "vector_db"


logger = logging.getLogger("sync_vector_store")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronise processed chunks and vector DB from raw PDFs."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory with source PDF files (default: %(default)s)",
    )
    parser.add_argument(
        "--processed-json",
        type=Path,
        default=DEFAULT_PROCESSED_JSON,
        help="Destination JSON path for processed chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--vector-db",
        type=Path,
        default=DEFAULT_VECTOR_DB,
        help="Directory for the Chroma vector database (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-existing-vector",
        action="store_true",
        help="Do not wipe existing vector DB before rebuilding (default: wipe).",
    )
    return parser.parse_args()


def ensure_paths(raw_dir: Path, processed_json: Path, vector_db: Path) -> None:
    if not raw_dir.exists() or not any(raw_dir.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDF files found in raw directory: {raw_dir}. "
            "Add PDFs to DiaScreenRAG/data/raw and run the script again."
        )

    processed_json.parent.mkdir(parents=True, exist_ok=True)
    vector_db.parent.mkdir(parents=True, exist_ok=True)


def rebuild_processed_data(raw_dir: Path, processed_json: Path) -> None:
    logger.info("Processing PDFs from %s", raw_dir)
    process_pdfs_to_chunks(str(raw_dir), str(processed_json))
    logger.info("Processed chunks written to %s", processed_json)


def rebuild_vector_store(processed_json: Path, vector_db: Path, keep_existing: bool) -> None:
    if vector_db.exists() and not keep_existing:
        logger.info("Removing existing vector DB at %s", vector_db)
        shutil.rmtree(vector_db)

    logger.info("Rebuilding vector DB at %s", vector_db)
    create_vector_store(str(processed_json), str(vector_db))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = parse_args()

    ensure_paths(args.raw_dir, args.processed_json, args.vector_db)
    rebuild_processed_data(args.raw_dir, args.processed_json)
    rebuild_vector_store(args.processed_json, args.vector_db, args.keep_existing_vector)

    logger.info("Sync complete.")


if __name__ == "__main__":
    main()

