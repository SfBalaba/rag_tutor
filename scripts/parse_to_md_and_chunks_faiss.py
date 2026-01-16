import os
import re
import shutil
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from google.colab import userdata


DATA_DIR = Path("drive/MyDrive/data")
FULL_MD_OUTPUT_DIR = Path("drive/MyDrive/try_2/parsed_full")
CHUNKS_OUTPUT_DIR = Path("drive/MyDrive/try_2/parsed_chunks")
FAISS_OUTPUT_DIR = Path("drive/MyDrive/try_2/faiss_db")

LEVELS = ["elementary", "middle_school", "high_school", "university"]

USER_PROMPT = (
    "Проанализируй документ максимально точно. Требования:\n"
    "1. Сохрани полную логическую структуру: заголовки, подзаголовки, списки, абзацы.\n"
    "2. Все математические формулы должны быть сохранены в корректном LaTeX:\n"
    "   - Inline: $...$\n"
    "   - Display: $$...$$\n"
    "3. Таблицы — в читаемом markdown-формате.\n"
    "4. Изображения — отметь как ![Изображение](image_X.png) или <!-- Изображение: описание -->.\n"
    "5. Текст может содержать р Русский, латиницу, греческие буквы, римские цифры — сохрани как есть.\n"
    "6. НЕ добавляй пояснений, преамбул или обёрток. Только чистый markdown.\n"
    "7. Если формула не распознана — оставь как [формула не распознана], но НЕ искажай текст."
)


parser = LlamaParse(
    api_key=userdata.get("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    language="ru",
    user_prompt=USER_PROMPT,
    show_progress=False,
    ignore_errors=False,
    region="eu", 
)


async def safe_load_data(file_path: str):
    return await parser.aload_data(file_path)

# Глобальный loop — создаём один раз
_GLOBAL_LOOP = None

def get_or_create_event_loop():
    global _GLOBAL_LOOP
    if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
        _GLOBAL_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_GLOBAL_LOOP)
    return _GLOBAL_LOOP

def parse_with_llamaparse(file_path: str):
    loop = get_or_create_event_loop()
    try:
        return loop.run_until_complete(safe_load_data(file_path))
    except Exception as e:
        # Пересоздаём loop при фатальных ошибках
        if "different event loop" in str(e) or "closed" in str(e).lower():
            loop.close()
            _GLOBAL_LOOP = None
            loop = get_or_create_event_loop()
            return loop.run_until_complete(safe_load_data(file_path))
        else:
            raise


def convert_djvu_to_pdf_fallback(djvu_path: Path) -> Path | None:
    if not shutil.which("ebook-convert"):
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = Path(tmp.name)
        subprocess.run(
            ["ebook-convert", str(djvu_path), str(tmp_pdf)],
            capture_output=True, timeout=600, check=True
        )
        return tmp_pdf
    except Exception as e:
        print(f"  ⚠️ ebook-convert fallback failed: {e}")
        return None

def convert_djvu_to_pdf(djvu_path: Path) -> Path | None:
    if not shutil.which("ddjvu"):
        print(f"  ⚠️ 'ddjvu' не найден.")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = Path(tmp.name)
        result = subprocess.run(
            ["ddjvu", "-format=pdf", str(djvu_path), str(tmp_pdf)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  ⚠️ Ошибка ddjvu: {result.stderr[:200]}")
            tmp_pdf.unlink(missing_ok=True)
            fallback = convert_djvu_to_pdf_fallback(djvu_path)
            return fallback
        return tmp_pdf
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Таймаут ddjvu: {djvu_path.name}")
        tmp_pdf.unlink(missing_ok=True)
        return convert_djvu_to_pdf_fallback(djvu_path)
    except Exception as e:
        print(f"  ❌ Ошибка конвертации {djvu_path.name}: {e}")
        return None

def parse_document(file_path: Path) -> str | None:
    ext = file_path.suffix.lower()
    use_path = file_path

    if ext == ".djvu":
        tmp_pdf = convert_djvu_to_pdf(file_path)
        if tmp_pdf is None:
            return None
        use_path = tmp_pdf
    elif ext != ".pdf":
        return None

    try:
        documents = parse_with_llamaparse(str(use_path))
        if not documents:
            return None
        return "\n\n---\n\n".join([doc.text for doc in documents])
    except Exception as e:
        print(f"  ❌ Ошибка парсинга {file_path.name}: {e}")
        return None
    finally:
        if ext == ".djvu" and use_path != file_path:
            use_path.unlink(missing_ok=True)


def split_markdown_hierarchical(md_text: str) -> List[Dict]:
    lines = md_text.split('\n')
    stack = []  # [(level, heading_text), ...]
    chunks = []
    current_content_lines = []

    heading_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    def flush_chunk():
        if not current_content_lines:
            return
        content = "\n".join(current_content_lines).strip()
        if not content or all(line.strip().startswith('#') for line in current_content_lines if line.strip()):
            return

        heading_path = [item[1] for item in stack] if stack else ["Document_Root"]
        current_heading = heading_path[-1] if heading_path else "Document_Root"

        chunks.append({
            "heading_path": heading_path,
            "heading": current_heading,
            "content": content,
            "level_stack": [item[0] for item in stack] if stack else [1]
        })

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            flush_chunk()
            current_content_lines = [line]
            new_level = len(match.group(1))
            new_heading = match.group(2).strip()
            while stack and stack[-1][0] >= new_level:
                stack.pop()
            stack.append((new_level, new_heading))
        else:
            current_content_lines.append(line)

    flush_chunk()
    return chunks

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s\-]', '', name.strip())
    name = re.sub(r'\s+', '_', name)
    return name[:100] or "unnamed"

def save_chunks_legacy(chunks: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        heading = chunk["heading"]
        fname = f"{i:03d}_{sanitize_filename(heading)}.md"
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            f.write(chunk["content"])


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

def embed_and_save_to_faiss(all_chunks_with_meta: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⚙️ Загрузка модели эмбеддингов...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')

    texts = []
    metas = []
    for chunk in all_chunks_with_meta:
        texts.append("passage: " + chunk["content"])
        metas.append({
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "level": chunk["level"],
            "book_title": chunk["book_title"],
            "grade": chunk["grade"],
            "heading_path": chunk["heading_path"]
        })

    print(f"⚙️ Генерация эмбеддингов для {len(texts)} чанков...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # <-- Важно: нормализуем сразу
    )

    # Убедимся, что тип float32
    embeddings = embeddings.astype('float32')

    # Создаём индекс на основе скалярного произведения (эквивалент косинусу для нормированных векторов)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # <-- Используем IndexFlatIP
    index.add(embeddings)

    faiss.write_index(index, str(output_dir / "faiss.index"))

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    np.save(output_dir / "embeddings.npy", embeddings)
    print(f"✅ FAISS (косинусная близость) и метаданные сохранены в {output_dir}")



def main():
    print("🚀 Начинаю парсинг, чанкизацию и векторизацию...")

    all_chunks_with_meta = []
    chunk_counter = 0

    for level in LEVELS:
        level_dir = DATA_DIR / level
        if not level_dir.exists():
            continue

        file_paths = []
        for ext in [".pdf", ".djvu"]:
            file_paths.extend(level_dir.rglob(f"*{ext}"))
        file_paths = [f for f in file_paths if f.is_file() and not f.name.startswith(".")]

        if not file_paths:
            continue

        print(f"\n📂 Уровень: {level} ({len(file_paths)} файлов)")

        for file_path in tqdm(file_paths, desc=f"  {level}"):
            if "checkpoint" in file_path.name:
                continue

            md_content = parse_document(file_path)
            if md_content is None:
                continue

            # Сохранение полного MD
            rel_path = file_path.relative_to(DATA_DIR)
            full_md_path = FULL_MD_OUTPUT_DIR / rel_path.with_suffix(".md")
            full_md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            # Чанки
            chunks = split_markdown_hierarchical(md_content)
            if not chunks:
                continue

            # Метаданные из пути
            book_title = rel_path.stem
            grade = rel_path.parent.name  

            for chunk in chunks:
                all_chunks_with_meta.append({
                    "chunk_id": chunk_counter,
                    "source_file": str(rel_path),
                    "level": level,
                    "book_title": book_title,
                    "grade": grade,
                    **chunk
                })
                chunk_counter += 1

            # Сохранение чанков в файлы (для отладки)
            chunks_dir = CHUNKS_OUTPUT_DIR / rel_path.parent / rel_path.stem
            save_chunks_legacy(chunks, chunks_dir)

    # Векторизация
    if all_chunks_with_meta:
        embed_and_save_to_faiss(all_chunks_with_meta, FAISS_OUTPUT_DIR)
    else:
        print("⚠️ Нет чанков для векторизации!")

    print(f"\n✅ Готово! Всего чанков: {len(all_chunks_with_meta)}")

if __name__ == "__main__":
    main()