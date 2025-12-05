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

# Пути
DATA_DIR = Path("../data")
FULL_MD_OUTPUT_DIR = Path("../parsed_full")
CHUNKS_OUTPUT_DIR = Path("../parsed_chunks")

LEVELS = ["elementary", "middle_school", "high_school", "university"]

USER_PROMPT = (
    "Проанализируй документ максимально точно. Требования:\n"
    "1. Сохрани полную логическую структуру: заголовки, подзаголовки, списки, абзацы.\n"
    "2. Все математические формулы должны быть сохранены в корректном LaTeX:\n"
    "   - Inline: $...$\n"
    "   - Display: $$...$$\n"
    "3. Таблицы — в читаемом markdown-формате.\n"
    "4. Изображения — отметь как ![Изображение](image_X.png) или <!-- Изображение: описание -->.\n"
    "5. Текст может содержать русский, латиницу, греческие буквы, римские цифры — сохрани как есть.\n"
    "6. Не добавляй пояснений, преамбул или обёрток. Только чистый markdown."
)


parser = LlamaParse(
    api_key=userdata.get("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    language="ru",
    user_prompt=USER_PROMPT,
    show_progress=False,  # tqdm сам показывает прогресс
    ignore_errors=False,
)

# --- Асинхронная обёртка для безопасного вызова ---
async def safe_load_data(file_path: str):
    return await parser.aload_data(file_path)

def parse_with_llamaparse(file_path: str):
    """Выполняет асинхронный вызов в синхронном контексте безопасно."""
    try:
        # Используем существующий loop, если он работает
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop closed")
        return loop.run_until_complete(safe_load_data(file_path))
    except:
        # Если loop сломан — создаём новый
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(safe_load_data(file_path))
        loop.close()
        return result

# --- Конвертация DJVU ---
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
            return None
        return tmp_pdf
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Таймаут ddjvu: {djvu_path.name}")
        tmp_pdf.unlink(missing_ok=True)
        return None
    except Exception as e:
        print(f"  ❌ Ошибка конвертации {djvu_path.name}: {e}")
        return None

# --- Парсинг одного файла ---
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

# --- Разбиение на чанки ---
def split_markdown_by_headings(md_text: str) -> List[Dict[str, str]]:
    lines = md_text.split('\n')
    chunks = []
    current_chunk = {"heading": "Document_Start", "content": "", "level": 1}
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*)')

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            if current_chunk["content"].strip() or current_chunk["heading"] != "Document_Start":
                chunks.append(current_chunk.copy())
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            current_chunk = {
                "heading": heading_text,
                "content": line + "\n",
                "level": level
            }
        else:
            current_chunk["content"] += line + "\n"

    if current_chunk["content"].strip():
        chunks.append(current_chunk)
    return chunks

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s\-]', '', name.strip())
    name = re.sub(r'\s+', '_', name)
    return name[:100] or "unnamed"

def save_chunks(chunks: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        heading = chunk["heading"]
        fname = f"{i:03d}_{sanitize_filename(heading)}.md"
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            f.write(chunk["content"])

# --- Основной цикл ---
def main():
    print("🚀 Начинаю парсинг...")

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

            rel_path = file_path.relative_to(DATA_DIR)

            full_md_path = FULL_MD_OUTPUT_DIR / rel_path.with_suffix(".md")
            full_md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            chunks = split_markdown_by_headings(md_content)
            chunks_dir = CHUNKS_OUTPUT_DIR / rel_path.parent / rel_path.stem
            save_chunks(chunks, chunks_dir)

            print(f"    ✅ {rel_path.name}")

    print(f"\n✅ Готово!")

if __name__ == "__main__":
    main()