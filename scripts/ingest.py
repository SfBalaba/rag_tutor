# scripts/ingest.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


DATA_DIR = Path("../data")
VECTOR_DB_PATH = Path("../vectorbase")
SAMPLES_PATH = Path("../samples/sample_chunks.json")

LEVELS = ["elementary", "middle_school", "high_school", "university"]

CHUNK_PARAMS = {
    "elementary": {"chunk_size": 300, "chunk_overlap": 50},
    "middle_school": {"chunk_size": 400, "chunk_overlap": 60},
    "high_school": {"chunk_size": 500, "chunk_overlap": 80},
    "university": {"chunk_size": 700, "chunk_overlap": 100},
}

# Загружаем локальную модель эмбеддингов
embedding_model = SentenceTransformer("/home/sofya/all-MiniLM-L6-v2")



def extract_grade_from_path(file_path: Path, level_dir: Path) -> str:
    try:
        rel_parts = file_path.relative_to(level_dir).parts
        for part in rel_parts:
            if "класс" in part or "course" in part.lower():
                match = re.search(r'(\d+)', part)
                if match:
                    return match.group(1)
    except ValueError:
        pass
    return "general"

def get_all_document_files(base_dir: Path) -> List[Path]:
    supported_ext = {".pdf", ".doc", ".docx", ".djvu"}
    files = []
    for file_path in base_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_ext:
            if file_path.name.startswith("."):
                continue
            files.append(file_path)
    return files


try:
    from pdf2image import convert_from_path
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    print("⚠️  Установите зависимости для OCR: pip install pdf2image pytesseract")
    TESSERACT_AVAILABLE = False

def ocr_pdf_to_text(pdf_path: Path, lang: str = "rus") -> List[str]:
    """Конвертирует PDF в текст через OCR (одна строка на страницу)."""
    if not TESSERACT_AVAILABLE:
        return []
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
        texts = []
        for img in images:
            text = pytesseract.image_to_string(img, lang=lang)
            texts.append(text)
        return texts
    except Exception as e:
        print(f"  ❌ OCR ошибка: {e}")
        return []

def load_document(file_path: Path):
    """Загружает документ с поддержкой OCR для сканов."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        # Сначала пробуем извлечь текст напрямую
        docs = PyMuPDFLoader(str(file_path)).load()
        # Если текст пустой — применяем OCR
        if not any(doc.page_content.strip() for doc in docs):
            print(f"  ⚠️ PDF {file_path.name} — пустой текст. Применяю OCR...")
            ocr_texts = ocr_pdf_to_text(file_path)
            docs = [
                Document(page_content=text, metadata={"source": str(file_path), "page": i})
                for i, text in enumerate(ocr_texts) if text.strip()
            ]
        return docs

    elif ext in (".doc", ".docx"):
        return UnstructuredWordDocumentLoader(str(file_path)).load()

    elif ext == ".djvu":
        if not shutil.which("ddjvu"):
            print("  ⚠️ Утилита 'ddjvu' не найдена. Установите пакет 'djvulibre'.")
            return []
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_pdf = Path(tmp.name)
            result = subprocess.run(
                ["ddjvu", "-format=pdf", str(file_path), str(tmp_pdf)],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode != 0:
                print(f"  ⚠️ Ошибка ddjvu: {result.stderr}")
                tmp_pdf.unlink(missing_ok=True)
                return []
            # Применяем OCR к конвертированному PDF
            print(f"  ⚠️ рПрименяем OCR к конвертированному PDF {tmp_pdf.name}. Применяю OCR...")
            
            ocr_texts = ocr_pdf_to_text(tmp_pdf)
            tmp_pdf.unlink(missing_ok=True)
            docs = [
                Document(page_content=text, metadata={"source": str(file_path), "page": i})
                for i, text in enumerate(ocr_texts) if text.strip()
            ]
            return docs
        except Exception as e:
            print(f"  ❌ Ошибка обработки DJVU {file_path.name}: {e}")
            tmp_pdf.unlink(missing_ok=True)
            return []

    else:
        return []



def main():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    all_sample_chunks = []

    for level in LEVELS:
        level_dir = DATA_DIR / level
        if not level_dir.exists():
            print(f"⚠️ Уровень '{level}' отсутствует. Пропускаем.")
            continue

        print(f"\n📂 Обработка уровня: {level}")
        file_paths = get_all_document_files(level_dir)
        print(f"  Найдено файлов: {len(file_paths)}")

        if not file_paths:
            continue

        params = CHUNK_PARAMS[level]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_texts, all_metadatas, all_ids = [], [], []

        for file_path in tqdm(file_paths, desc=f"  Загрузка ({level})"):
            if "checkpoint" in file_path.name:
                continue
            try:
                documents = load_document(file_path)
                if not documents:
                    continue

                chunks = text_splitter.split_documents(documents)
                grade = extract_grade_from_path(file_path, level_dir)
                source_rel = str(file_path.relative_to(DATA_DIR))

                for i, chunk in enumerate(chunks):
                    text = chunk.page_content.strip()
                    if not text:
                        continue
                    meta = {
                        "level": level,
                        "grade": grade,
                        "source": source_rel,
                        "filename": file_path.name,
                    }
                    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                        meta.update({
                            k: v for k, v in chunk.metadata.items()
                            if isinstance(v, (str, int, float, bool)) and k not in meta
                        })
                    chunk_id = f"{level}_{file_path.stem}_{i}"
                    all_texts.append(text)
                    all_metadatas.append(meta)
                    all_ids.append(chunk_id)

                    if len(all_sample_chunks) < 10:
                        all_sample_chunks.append({
                            "id": chunk_id,
                            "text": text[:200] + "..." if len(text) > 200 else text,
                            "metadata": meta
                        })

            except Exception as e:
                print(f"\n  ❌ Ошибка обработки {file_path}: {e}")
                continue

        if not all_texts:
            print(f"  ⚠️ Нет чанков для уровня '{level}'")
            continue

        # Сохраняем в Chroma (один раз!)
        collection = client.get_or_create_collection(name=level, embedding_function=None)
        embeddings = embedding_model.encode(all_texts, convert_to_numpy=True).tolist()
        collection.add(
            documents=all_texts,
            metadatas=all_metadatas,
            embeddings=embeddings,
            ids=all_ids
        )
        print(f"  ✅ Уровень '{level}' сохранён ({len(all_texts)} чанков).")

    # Сохраняем сэмплы
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_sample_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Ингест завершён. Сэмплы: {SAMPLES_PATH}")

if __name__ == "__main__":
    main()