import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from langchain.text_splitter import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from markdownify import markdownify
from tqdm import tqdm

# Добавление корневой директории проекта в PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорты из core модулей должны быть после добавления в sys.path
from core.config import config  # noqa: E402
from core.vector_store import get_embedding_model  # noqa: E402


def preprocess_article(content: str) -> str:
    """Предобработка статьи: HTML -> Markdown + очистка"""
    if not content or pd.isna(content):
        return ""

    # Конвертация HTML в Markdown
    markdown_content = markdownify(content, heading_style="ATX")

    # Очистка специальных тегов ТЖ
    special_tags = [
        "[author]",
        "[/author]",
        "[img]",
        "[/img]",
        "[nobr]",
        "[/nobr]",
        "[quote]",
        "[/quote]",
        "[video]",
        "[/video]",
        "[audio]",
        "[/audio]",
    ]

    for tag in special_tags:
        markdown_content = markdown_content.replace(tag, "")

    # Нормализация пробелов
    markdown_content = markdown_content.replace("\u00a0", " ")  # неразрывный пробел
    markdown_content = "\n".join(line.strip() for line in markdown_content.split("\n"))

    return markdown_content.strip()


def create_chunker(chunker_type: str, chunk_size: int, chunk_overlap: int):
    """Создает чанкер указанного типа"""
    if chunker_type == "markdown":
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunker_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
        )
    elif chunker_type == "sentence":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[". ", "! ", "? ", "\n\n", "\n", " "],
        )
    elif chunker_type == "token":
        return SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size // 4,  # примерно 4 символа на токен
        )
    elif chunker_type == "hierarchical":
        # Двухуровневое разбиение: сначала крупные чанки, потом мелкие
        return HierarchicalChunker(
            primary_size=chunk_size * 2, secondary_size=chunk_size, overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Неизвестный тип чанкера: {chunker_type}")


class HierarchicalChunker:
    """Иерархический чанкер: двухуровневое разбиение"""

    def __init__(self, primary_size: int = 2000, secondary_size: int = 600, overlap: int = 100):
        self.primary_splitter = MarkdownTextSplitter(chunk_size=primary_size, chunk_overlap=overlap)
        self.secondary_splitter = MarkdownTextSplitter(
            chunk_size=secondary_size, chunk_overlap=overlap
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Разбивает документы иерархически"""
        result = []

        for doc in documents:
            # Первый уровень: крупные чанки
            primary_chunks = self.primary_splitter.split_documents([doc])

            for i, primary_chunk in enumerate(primary_chunks):
                # Второй уровень: мелкие чанки из крупных
                secondary_chunks = self.secondary_splitter.split_documents([primary_chunk])

                for j, secondary_chunk in enumerate(secondary_chunks):
                    # Добавляем метаданные об иерархии
                    secondary_chunk.metadata.update(
                        {
                            "primary_chunk_id": i,
                            "secondary_chunk_id": j,
                            "chunk_type": "hierarchical",
                        }
                    )
                    result.append(secondary_chunk)

        return result


def process_data_to_documents(df: pd.DataFrame, text_splitter) -> list[Document]:
    """Обрабатывает DataFrame в список документов"""
    documents = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обработка статей"):
        # Создаем полный текст статьи
        title = str(row.get("title", "")).strip()
        subtitle = str(row.get("subtitle", "")).strip()
        content = str(row.get("content_raw", "")).strip()

        # Формируем полный текст
        full_text_parts = []
        if title and title != "nan":
            full_text_parts.append(f"# {title}")
        if subtitle and subtitle != "nan":
            full_text_parts.append(f"## {subtitle}")
        if content and content != "nan":
            full_text_parts.append(content)

        full_text = "\n\n".join(full_text_parts)

        if not full_text.strip():
            continue

        # Создаем документ
        doc = Document(
            page_content=full_text,
            metadata={
                "article_id": str(row.get("id", idx)),
                "title": title,
                "subtitle": subtitle,
                "rubrics": str(row.get("rubrics", "")),
                "tags": str(row.get("tags", "")),
                "author": str(row.get("author", "")),
                "source_row": idx,
            },
        )

        documents.append(doc)

    # Разбиваем на чанки
    print(f"Разбиение {len(documents)} статей на чанки...")
    chunks = text_splitter.split_documents(documents)

    # Добавляем уникальные ID для чанков
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i}"

    print(f"Создано {len(chunks)} чанков")
    return chunks


def create_vectorstore(documents: list[Document], embedding_model) -> FAISS:
    """Создает векторное хранилище"""
    print("Создание векторного хранилища...")

    # Создаем FAISS индекс
    vectorstore = FAISS.from_documents(documents, embedding_model)

    print(f"Векторное хранилище создано с {len(documents)} документами")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, output_path: str):
    """Сохраняет векторное хранилище"""
    os.makedirs(output_path, exist_ok=True)
    vectorstore.save_local(output_path)
    print(f"Векторное хранилище сохранено в {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Создание векторной базы данных для ТЖ")

    parser.add_argument(
        "--input", type=str, required=True, help="Путь к входному CSV файлу с данными ТЖ"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Путь к выходной директории для векторной БД"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=config["embedding"]["model"],
        help="Модель для создания эмбеддингов",
    )
    parser.add_argument(
        "--chunker-type",
        type=str,
        choices=["markdown", "recursive", "sentence", "token", "hierarchical"],
        default="markdown",
        help="Тип чанкера для разбиения текста",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1200, help="Размер чанка при разбиении текста"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=0, help="Перекрытие чанков при разбиении текста"
    )
    parser.add_argument(
        "--filter-ugc",
        action="store_true",
        default=True,
        help="Фильтровать материалы из сообщества (UGC)",
    )
    parser.add_argument("--separator", type=str, default=";", help="Разделитель в CSV файле")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Загрузка данных из {args.input}...")
    df = pd.read_csv(args.input, sep=args.separator)
    print(f"Загружено {len(df)} записей")

    # Удаляем записи без контента
    initial_count = len(df)
    df = df.dropna(subset=["content_raw"])
    print(f"После удаления пустых записей: {len(df)} ({initial_count - len(df)} удалено)")

    # Фильтрация UGC материалов
    if args.filter_ugc and "rubrics" in df.columns:
        initial_count = len(df)
        df = df[df["rubrics"] != "Материалы из сообщества"]
        print(f"После фильтрации UGC: {len(df)} ({initial_count - len(df)} удалено)")

    # Предобработка контента
    print("Предобработка статей...")
    df["content_raw"] = df["content_raw"].apply(preprocess_article)

    # Создание чанкера
    print(f"Создание чанкера типа: {args.chunker_type}")
    text_splitter = create_chunker(args.chunker_type, args.chunk_size, args.chunk_overlap)

    # Обработка в документы
    documents = process_data_to_documents(df, text_splitter)

    # Создание embedding модели
    print(f"Загрузка embedding модели: {args.embedding_model}")
    embedding_model = get_embedding_model(model_name=args.embedding_model)

    # Создание векторного хранилища
    vectorstore = create_vectorstore(documents, embedding_model)

    # Сохранение
    save_vectorstore(vectorstore, args.output)

    print("✅ Векторная база данных успешно создана!")
    print("📊 Статистика:")
    print(f"   - Обработано статей: {len(df)}")
    print(f"   - Создано чанков: {len(documents)}")
    print(f"   - Тип чанкера: {args.chunker_type}")
    print(f"   - Размер чанка: {args.chunk_size}")
    print(f"   - Перекрытие: {args.chunk_overlap}")
    print(f"   - Embedding модель: {args.embedding_model}")
    print(f"   - Сохранено в: {args.output}")


if __name__ == "__main__":
    main()
