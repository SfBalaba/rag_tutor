#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.config import config
from core.llm import get_llm
from core.ranking import get_doc_content, rerank_documents
from core.vector_store import get_vector_store


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Генерация синтетических ground truth данных")

    parser.add_argument(
        "--questions-csv",
        type=str,
        required=True,
        help="Путь к CSV файлу с вопросами (колонка 'question')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="research/data/synthetic_qa.csv",
        help="Путь к выходному CSV файлу",
    )
    parser.add_argument(
        "--num-examples", type=int, default=100, help="Количество примеров для генерации"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=5, help="Количество чанков для каждого вопроса"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Температура модели для генерации ответов"
    )

    return parser.parse_args()


def get_relevant_chunks_for_query(vector_store, query: str, n: int = 5):
    """Получает релевантные чанки для запроса"""
    # Получаем больше документов чем нужно для лучшего отбора
    search_k = min(n * 3, config.get("database", {}).get("search_top_k", 20))
    docs = vector_store.similarity_search(query, k=search_k)

    # Применяем реранкер если включен
    use_reranker = config.get("reranker", {}).get("enabled", False)
    if use_reranker and docs:
        docs = rerank_documents(query, docs)

    # Берем топ-n документов
    return docs[:n]


def create_synthetic_dataset(
    questions_df: pd.DataFrame,
    num_examples: int = 100,
    num_chunks: int = 5,
    temperature: float = 0.1,
):
    """Создает синтетический датасет вопрос-ответ"""

    # Инициализация компонентов
    vector_store = get_vector_store()
    llm = get_llm(temperature=temperature)

    # Создаем промпт для генерации ответа
    qa_prompt = config.get("qa_prompt", "Контекст: {context}\n\nВопрос: {question}\n\nОтвет:")
    prompt = ChatPromptTemplate.from_template(qa_prompt)

    qa_chain = prompt | llm | StrOutputParser()

    dataset = []

    # Получаем уникальные вопросы
    if "question" not in questions_df.columns:
        raise ValueError("CSV файл должен содержать колонку 'question'")

    unique_questions = questions_df["question"].dropna().unique()
    questions_to_use = (
        pd.Series(unique_questions)
        .sample(min(len(unique_questions), num_examples), random_state=42)
        .tolist()
    )

    print(f"📝 Генерируем ответы для {len(questions_to_use)} вопросов...")

    for question in tqdm(questions_to_use, desc="Генерация датасета"):
        try:
            # Получаем релевантные чанки
            selected_chunks = get_relevant_chunks_for_query(vector_store, question, num_chunks)

            if not selected_chunks:
                print(f"⚠️ Не найдены документы для вопроса: {question}")
                continue

            # Форматируем контекст
            docs_separator = config.get("docs_separator", "\n\n-----")
            context = docs_separator.join(get_doc_content(chunk) for chunk in selected_chunks)

            # Генерируем ответ
            answer = qa_chain.invoke({"question": question, "context": context})

            dataset.append(
                {
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "num_chunks": len(selected_chunks),
                }
            )

        except Exception as e:
            print(f"⚠️ Ошибка при обработке вопроса '{question}': {e}")
            continue

    print(f"✅ Создано {len(dataset)} примеров")
    return dataset


def validate_dataset(dataset: list):
    """Валидация созданного датасета"""
    if not dataset:
        raise ValueError("Датасет пуст")

    # Проверяем наличие обязательных полей
    required_fields = ["question", "answer", "context"]
    for i, item in enumerate(dataset):
        for field in required_fields:
            if field not in item or not item[field].strip():
                print(f"⚠️ Предупреждение: пустое поле '{field}' в примере {i + 1}")

    # Статистика
    avg_answer_len = sum(len(item["answer"]) for item in dataset) / len(dataset)
    avg_context_len = sum(len(item["context"]) for item in dataset) / len(dataset)

    print("\n📊 Статистика датасета:")
    print(f"  Примеров: {len(dataset)}")
    print(f"  Средняя длина ответа: {avg_answer_len:.0f} символов")
    print(f"  Средняя длина контекста: {avg_context_len:.0f} символов")


def main():
    """Главная функция"""
    args = parse_args()

    print("🚀 Генерация синтетических ground truth данных")
    print(f"Входной файл: {args.questions_csv}")
    print(f"Выходной файл: {args.output_file}")

    # Создаем директорию для выходного файла
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Загружаем файл с вопросами
    if not os.path.exists(args.questions_csv):
        raise FileNotFoundError(f"Файл не найден: {args.questions_csv}")

    try:
        questions_df = pd.read_csv(args.questions_csv)
    except Exception as e:
        raise ValueError(f"Ошибка чтения CSV файла: {e}") from e

    print(f"📄 Загружено {len(questions_df)} записей из {args.questions_csv}")

    # Фильтрация (если есть колонка rubrics, исключаем "Материалы из сообщества")
    if "rubrics" in questions_df.columns:
        initial_count = len(questions_df)
        questions_df = questions_df[questions_df["rubrics"] != "Материалы из сообщества"]
        filtered_count = len(questions_df)
        if initial_count != filtered_count:
            print(
                f"🔍 Исключены записи с рубрикой 'Материалы из сообщества': {initial_count} -> {filtered_count}"
            )

    # Создание синтетического датасета
    dataset = create_synthetic_dataset(
        questions_df,
        num_examples=args.num_examples,
        num_chunks=args.num_chunks,
        temperature=args.temperature,
    )

    # Валидация
    validate_dataset(dataset)

    # Сохранение результатов
    df_output = pd.DataFrame(dataset)
    df_output.to_csv(args.output_file, index=False)

    print(f"✅ Синтетический датасет сохранен: {args.output_file}")

    # Выводим примеры
    if len(dataset) > 0:
        print("\n📝 Пример сгенерированной записи:")
        example = dataset[0]
        print(f"Вопрос: {example['question'][:100]}...")
        print(f"Ответ: {example['answer'][:200]}...")


if __name__ == "__main__":
    main()
