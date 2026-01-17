import asyncio
import gc
import json
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False

try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from core.config import config
from core.llm import get_llm
from core.llm.deepeval_adapter import create_deepeval_adapter


@contextmanager
def gpu_memory_manager():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


_bleurt_model = None
_bleurt_tokenizer = None


def get_bleurt_model():
    global _bleurt_model
    if not BLEURT_AVAILABLE or _bleurt_model is not None:
        return _bleurt_model

    with gpu_memory_manager():
        device = (
            "cuda:1"
            if torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        _bleurt_model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20")
        _bleurt_model.eval()
        if torch.cuda.is_available():
            if device == "cuda:1":
                torch.cuda.set_per_process_memory_fraction(0.5, device=1)
            _bleurt_model = _bleurt_model.to(device)
    return _bleurt_model


def get_bleurt_tokenizer():
    global _bleurt_tokenizer
    if not BLEURT_AVAILABLE or _bleurt_tokenizer is not None:
        return _bleurt_tokenizer
    _bleurt_tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")
    return _bleurt_tokenizer


def split_docs(context):
    if isinstance(context, list):
        return context
    separator = config.get("docs_separator", "\n\n-----")
    return [doc.strip() for doc in context.split(separator) if doc.strip()]


def create_test_cases(
    dataset: pd.DataFrame,
    system_responses: pd.DataFrame | None = None,
    limit: int | None = None,
    actual_contexts: list[str] | None = None,
) -> list["LLMTestCase"]:
    """Создает тестовые случаи для оценки"""
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval не установлен. Выполните: pip install deepeval")

    if limit is not None and limit < len(dataset):
        if system_responses is not None:
            # Сохраняем консистентность индексов при выборке
            dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)
            system_responses = system_responses.reset_index(drop=True)

            # Обеспечиваем, что размеры совпадают
            if len(dataset) != len(system_responses):
                min_len = min(len(dataset), len(system_responses))
                dataset = dataset.iloc[:min_len]
                system_responses = system_responses.iloc[:min_len]

            # Если предоставлены актуальные контексты, также обрезаем их
            if actual_contexts is not None:
                if len(actual_contexts) != len(dataset):
                    min_len = min(len(dataset), len(actual_contexts))
                    actual_contexts = actual_contexts[:min_len]
        else:
            dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)

    test_cases = []
    for i, (_, row) in enumerate(
        tqdm(dataset.iterrows(), total=len(dataset), desc="Создание тестовых случаев")
    ):
        # Если есть ответы системы, используем их для actual_output
        if system_responses is not None:
            actual_output = system_responses.iloc[i].get(
                "system_answer", system_responses.iloc[i].get("answer", "")
            )
        else:
            # Иначе используем ответ из датасета
            actual_output = row["answer"]

        # Определяем контекст для retrieval_context
        if actual_contexts is not None:
            context_to_use = actual_contexts[i]
        else:
            context_to_use = row["context"]

        # Создаем тестовый случай
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=actual_output,
            expected_output=row["answer"],
            retrieval_context=split_docs(context_to_use),
        )
        test_cases.append(test_case)

    return test_cases


# Функция удалена - используется create_deepeval_adapter из core.llm.deepeval_adapter


def evaluate_dataset(
    dataset: pd.DataFrame,
    system_responses: pd.DataFrame | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    limit: int | None = None,
    actual_contexts: list[str] | None = None,
) -> pd.DataFrame:
    """Оценивает датасет с использованием различных метрик"""

    if model_name is None:
        model_name = config["model"]["name"]

    # Создаем тестовые случаи
    test_cases = create_test_cases(dataset, system_responses, limit, actual_contexts)

    # Создаем модель для оценки (если DeepEval доступен)
    eval_model = None
    if DEEPEVAL_AVAILABLE:
        eval_model = create_deepeval_adapter(model_name, temperature)

    # Подготовка данных для BLEURT и косинусных метрик
    references = [test_case.expected_output for test_case in test_cases]
    candidates = [test_case.actual_output for test_case in test_cases]

    # Расчет BLEURT оценок
    bleurt_scores = calculate_bleurt_score(references, candidates)

    # Расчет косинусных оценок (batch версия для оптимизации)
    cosine_scores = calculate_cosine_similarity_batch(references, candidates)

    # Инициализируем метрики DeepEval (если доступны)
    deepeval_metrics = []
    if DEEPEVAL_AVAILABLE and eval_model:
        deepeval_metrics = [
            FaithfulnessMetric(threshold=0.5, model=eval_model),
            AnswerRelevancyMetric(threshold=0.5, model=eval_model),
            ContextualRelevancyMetric(threshold=0.5, model=eval_model),
            GEval(
                name="Correctness",
                criteria="Определите, является ли 'фактический вывод' правильным на основе 'ожидаемого вывода'.",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.5,
                model=eval_model,
            ),
        ]

    # Счетчики для диагностики
    metric_error_counts = (
        {metric.__class__.__name__.replace("Metric", ""): 0 for metric in deepeval_metrics}
        if deepeval_metrics
        else {}
    )

    results = []
    for i, test_case in enumerate(tqdm(test_cases, desc="Оценка примеров")):
        metric_scores = {}

        # Оценка DeepEval метриками
        for metric in deepeval_metrics:
            metric_name = metric.__class__.__name__.replace("Metric", "")
            try:
                metric.measure(test_case)
                metric_scores[metric_name] = metric.score
            except Exception as e:
                print(f"⚠️ Ошибка в метрике {metric_name}: {e}")
                metric_scores[metric_name] = 0.0
                metric_error_counts[metric_name] += 1

        # Добавляем BLEURT и косинусные оценки
        metric_scores["BLEURT"] = bleurt_scores[i]
        metric_scores["CosineSimilarity"] = cosine_scores[i]

        # Вычисляем общую оценку (среднее по всем метрикам)
        all_scores = list(metric_scores.values())
        avg_score = np.mean(all_scores) if all_scores else 0.0

        # Создаем запись результата
        result = {
            "question": test_case.input,
            "expected_output": test_case.expected_output,
            "actual_output": test_case.actual_output,
            "avg_score": avg_score,
            **metric_scores,
        }

        results.append(result)

    # Выводим диагностику ошибок
    if metric_error_counts:
        print("\n📊 Статистика ошибок метрик:")
        for metric_name, error_count in metric_error_counts.items():
            if error_count > 0:
                print(f"  {metric_name}: {error_count}/{len(test_cases)} ошибок")

    return pd.DataFrame(results)


def generate_report(evaluated_df: pd.DataFrame) -> dict[str, Any]:
    """Генерирует детальный отчет по результатам оценки"""
    # Определяем колонки с метриками (исключаем текстовые поля)
    exclude_columns = [
        "question",
        "expected_output",
        "actual_output",
        "system_answer",
        "golden_answer",
        "context",
        "chunk_ids",
    ]
    metric_columns = [col for col in evaluated_df.columns if col not in exclude_columns]

    report = {}

    print("\n📊 ДЕТАЛЬНАЯ СТАТИСТИКА ПО МЕТРИКАМ:")
    print(f"   Общее количество примеров: {len(evaluated_df)}")

    # Статистика по каждой метрике
    for metric in metric_columns:
        if metric in evaluated_df.columns:
            all_values = evaluated_df[metric]
            scores = all_values.dropna().tolist()
            none_count = all_values.isna().sum()

            print(f"\n   {metric}:")
            print(f"     Успешных оценок: {len(scores)}/{len(evaluated_df)}")
            print(f"     None значений: {none_count}")

            if scores:
                mean_val = float(np.mean(scores))
                std_val = float(np.std(scores))
                min_val = float(np.min(scores))
                max_val = float(np.max(scores))
                median_val = float(np.median(scores))

                report[metric] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "median": median_val,
                    "count": len(scores),
                    "none_count": int(none_count),
                }

                print(f"     Среднее: {mean_val:.4f} ± {std_val:.4f}")
                print(f"     Диапазон: [{min_val:.4f}, {max_val:.4f}]")
                print(f"     Медиана: {median_val:.4f}")

                # Дополнительная диагностика для проблемных метрик
                if metric in ["ContextualRelevancy", "Faithfulness"]:
                    print(f"     🔍 Детальная диагностика {metric}:")
                    if std_val > 0:
                        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                        print(f"       Коэффициент вариации: {cv:.1f}%")
                    if len(scores) > 5:
                        print(f"       Первые 5 значений: {scores[:5]}")
                        print(f"       Последние 5 значений: {scores[-5:]}")
            else:
                report[metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "count": 0,
                    "none_count": int(none_count),
                }

    # Общая статистика
    report["summary"] = {"total_examples": len(evaluated_df), "metrics_count": len(metric_columns)}

    # Если есть avg_score, добавляем его в сводку
    if "avg_score" in evaluated_df.columns:
        avg_scores = evaluated_df["avg_score"].dropna()
        if len(avg_scores) > 0:
            overall_avg = float(avg_scores.mean())
            report["summary"]["overall_avg_score"] = overall_avg
            print(f"\n   📈 Общая средняя оценка: {overall_avg:.4f}")

    return report


def filter_best_examples(evaluated_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Фильтрует лучшие примеры на основе avg_score"""
    if "avg_score" not in evaluated_df.columns:
        print("⚠️ Колонка 'avg_score' не найдена, возвращаем весь датасет")
        return evaluated_df

    filtered_df = evaluated_df[evaluated_df["avg_score"] >= threshold]
    print(f"📊 Отфильтровано {len(filtered_df)}/{len(evaluated_df)} примеров (порог: {threshold})")

    return filtered_df


def save_results(dataset: pd.DataFrame, output_path: str, report: dict[str, Any] | None = None):
    """Сохраняет результаты оценки"""
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Сохраняем основные результаты
    dataset.to_csv(output_path, index=False)
    print(f"✅ Результаты сохранены: {output_path}")

    # Сохраняем отчет если предоставлен
    if report:
        report_path = output_path.replace(".csv", "_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ Отчет сохранен: {report_path}")


# Асинхронные версии функций
async def calculate_bleurt_score_async(references: list[str], candidates: list[str]) -> list[float]:
    """Асинхронно рассчитывает BLEURT оценку"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, calculate_bleurt_score, references, candidates)


async def calculate_cosine_similarity_async(texts1: list[str], texts2: list[str]) -> list[float]:
    """Асинхронно рассчитывает косинусную близость"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, calculate_cosine_similarity_batch, texts1, texts2)


async def evaluate_dataset_async(
    dataset: pd.DataFrame,
    system_responses: pd.DataFrame | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
    limit: int | None = None,
    max_concurrency: int = 6,
    actual_contexts: list[str] | None = None,
) -> pd.DataFrame:
    """Асинхронная версия evaluate_dataset с параллельной обработкой метрик"""

    if model_name is None:
        model_name = config["model"]["name"]

    # Создаем тестовые случаи
    test_cases = create_test_cases(dataset, system_responses, limit, actual_contexts)

    # Создаем модель для оценки (если DeepEval доступен)
    eval_model = None
    if DEEPEVAL_AVAILABLE:
        eval_model = create_deepeval_adapter(model_name, temperature)

    # Подготовка данных для BLEURT и косинусных метрик
    references = [test_case.expected_output for test_case in test_cases]
    candidates = [test_case.actual_output for test_case in test_cases]

    # Асинхронный расчет BLEURT и косинусных оценок
    bleurt_task = asyncio.create_task(calculate_bleurt_score_async(references, candidates))
    cosine_task = asyncio.create_task(calculate_cosine_similarity_async(references, candidates))

    # Инициализируем метрики DeepEval (если доступны)
    deepeval_metrics = []
    if DEEPEVAL_AVAILABLE and eval_model:
        deepeval_metrics = [
            FaithfulnessMetric(threshold=0.5, model=eval_model),
            AnswerRelevancyMetric(threshold=0.5, model=eval_model),
            ContextualRelevancyMetric(threshold=0.5, model=eval_model),
            GEval(
                name="Correctness",
                criteria="Определите, является ли 'фактический вывод' правильным на основе 'ожидаемого вывода'.",
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.5,
                model=eval_model,
            ),
        ]

    # Ограничиваем количество одновременных запросов к LLM
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_test_case(i: int, test_case) -> tuple[int, dict[str, Any]]:
        """Обрабатывает один тестовый случай со всеми метриками"""
        async with semaphore:
            metric_scores = {}

            # Асинхронно вычисляем метрики
            for metric in deepeval_metrics:
                metric_name = metric.__class__.__name__.replace("Metric", "")
                try:
                    # Проверяем наличие асинхронного метода a_measure
                    if hasattr(metric, "a_measure"):
                        await metric.a_measure(test_case)
                    else:
                        # Если асинхронного метода нет, запускаем синхронный в executor
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, metric.measure, test_case)

                    metric_scores[metric_name] = metric.score
                except Exception as e:
                    print(f"⚠️ Ошибка в метрике {metric_name}: {e}")
                    metric_scores[metric_name] = 0.0

            return i, metric_scores

    # Запускаем обработку всех тестовых случаев параллельно
    print(
        f"🚀 Запуск асинхронной оценки {len(test_cases)} примеров (concurrency: {max_concurrency})"
    )

    tasks = [process_test_case(i, test_case) for i, test_case in enumerate(test_cases)]
    metric_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Получаем результаты других асинхронных задач
    bleurt_scores = await bleurt_task
    cosine_scores = await cosine_task

    # Формируем итоговые результаты
    results = []
    metric_error_counts = {
        metric.__class__.__name__.replace("Metric", ""): 0 for metric in deepeval_metrics
    }

    for result in metric_results:
        if isinstance(result, Exception):
            print(f"⚠️ Ошибка при обработке тестового случая: {result}")
            continue

        i, metric_scores = result
        test_case = test_cases[i]

        # Подсчитываем ошибки
        for metric_name, score in metric_scores.items():
            if score == 0.0:  # Предполагаем, что 0.0 означает ошибку
                metric_error_counts[metric_name] += 1

        # Добавляем BLEURT и косинусные оценки
        metric_scores["BLEURT"] = bleurt_scores[i]
        metric_scores["CosineSimilarity"] = cosine_scores[i]

        # Вычисляем общую оценку (среднее по всем метрикам)
        all_scores = list(metric_scores.values())
        avg_score = np.mean(all_scores) if all_scores else 0.0

        # Создаем запись результата
        result_item = {
            "question": test_case.input,
            "expected_output": test_case.expected_output,
            "actual_output": test_case.actual_output,
            "avg_score": avg_score,
            **metric_scores,
        }

        results.append(result_item)

    # Выводим диагностику ошибок
    if metric_error_counts:
        print("\n📊 Статистика ошибок метрик (асинхронная версия):")
        for metric_name, error_count in metric_error_counts.items():
            if error_count > 0:
                print(f"  {metric_name}: {error_count}/{len(test_cases)} ошибок")

    return pd.DataFrame(results)


def cleanup_models():
    """Очищает загруженные модели из памяти"""
    global _bleurt_model, _bleurt_tokenizer

    if _bleurt_model is not None:
        del _bleurt_model
        _bleurt_model = None

    if _bleurt_tokenizer is not None:
        del _bleurt_tokenizer
        _bleurt_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    print("✅ Модели очищены из памяти")


def stop():
    """Сбрасывает все ресурсы для освобождения памяти"""
    cleanup_models()

    # Дополнительная очистка
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()
    print("✅ Все ресурсы освобождены")


class OpenRouterAdapter:
    """Упрощенный адаптер для DeepEval (устаревший, используйте create_deepeval_adapter)"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config["model"]["name"]

    def generate(self, prompt: str) -> str:
        try:
            llm = get_llm()
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            print(f"⚠️ Ошибка в OpenRouterAdapter: {e}")
            return ""

    def get_model_name(self) -> str:
        return self.model_name


def create_deepeval_metrics(model_name: str | None = None, temperature: float = 0.0):
    """Создает DeepEval метрики с использованием нового адаптера"""
    if not DEEPEVAL_AVAILABLE:
        print("⚠️ DeepEval не установлен")
        return []

    model = create_deepeval_adapter(model_name, temperature)
    if model is None:
        print("⚠️ Не удалось создать DeepEval адаптер")
        return []

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=model, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True),
        ContextualRelevancyMetric(threshold=0.7, model=model, include_reason=True),
    ]
    return metrics


def create_correctness_metric(model_name: str | None = None, temperature: float = 0.0):
    """Создает метрику Correctness с использованием нового адаптера"""
    if not DEEPEVAL_AVAILABLE:
        return None

    model = create_deepeval_adapter(model_name, temperature)
    if model is None:
        return None

    return GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK",
        ],
        evaluation_params=["actual output", "expected output"],
        model=model,
    )


def evaluate_rag_triad_single(
    question: str,
    answer: str,
    contexts: list[str],
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Оценивает одиночный пример по RAG Triad метрикам"""
    if not DEEPEVAL_AVAILABLE:
        return {"error": "DeepEval не доступен"}

    try:
        metrics = create_deepeval_metrics(model_name, temperature)
        if not metrics:
            return {"error": "Не удалось создать метрики"}

        test_case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)

        results = {}
        for metric in metrics:
            try:
                metric.measure(test_case)
                results[metric.__class__.__name__] = {
                    "score": metric.score,
                    "success": metric.success,
                    "reason": getattr(metric, "reason", None),
                }
            except Exception as e:
                results[metric.__class__.__name__] = {"error": str(e)}

        return results

    except Exception as e:
        return {"error": f"Ошибка оценки RAG Triad: {e}"}


def evaluate_correctness_single(
    question: str,
    answer: str,
    expected_answer: str,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Оценивает одиночный пример по метрике Correctness"""
    if not DEEPEVAL_AVAILABLE:
        return {"error": "DeepEval не доступен"}

    try:
        metric = create_correctness_metric(model_name, temperature)
        if not metric:
            return {"error": "Не удалось создать метрику"}

        test_case = LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )

        metric.measure(test_case)
        return {
            "Correctness": {
                "score": metric.score,
                "success": metric.success,
                "reason": getattr(metric, "reason", None),
            }
        }

    except Exception as e:
        return {"error": f"Ошибка оценки Correctness: {e}"}


def get_doc_content(doc) -> str:
    if hasattr(doc, "page_content"):
        content = doc.page_content
        if isinstance(content, list | dict):
            return str(content)
        return str(content)
    elif isinstance(doc, dict):
        content = doc.get("page_content") or doc.get("content") or doc.get("text")
        if isinstance(content, list | dict):
            return str(content)
        return str(content) if content else ""
    return str(doc)


def load_bleurt_model(model_name: str = "lucadiliello/BLEURT-20"):
    if not BLEURT_AVAILABLE:
        return None, None

    try:
        config_bleurt = BleurtConfig.from_pretrained(model_name)
        model = BleurtForSequenceClassification.from_pretrained(model_name, config=config_bleurt)
        tokenizer = BleurtTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Ошибка загрузки BLEURT: {e}")
        return None, None


def calculate_bleurt_score(
    references: list[str], candidates: list[str], model_name: str = "lucadiliello/BLEURT-20"
) -> list[float]:
    """Рассчитывает BLEURT оценку с batch processing для оптимизации"""
    if not BLEURT_AVAILABLE:
        print("⚠️ BLEURT недоступен, возвращаем нулевые оценки")
        return [0.0] * len(candidates)

    model, tokenizer = load_bleurt_model(model_name)
    if model is None or tokenizer is None:
        return [0.0] * len(candidates)

    try:
        with gpu_memory_manager():
            with torch.no_grad():
                # Batch processing для экономии памяти
                batch_size = 8
                all_scores = []

                for i in range(0, len(references), batch_size):
                    batch_refs = references[i : i + batch_size]
                    batch_cands = candidates[i : i + batch_size]

                    inputs = tokenizer(
                        batch_refs,
                        batch_cands,
                        padding="longest",
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )

                    # Определяем device модели
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    batch_scores = model(**inputs).logits.flatten().cpu().tolist()
                    all_scores.extend(batch_scores)

                    # Очищаем промежуточную память
                    del inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                return all_scores
    except Exception as e:
        print(f"Ошибка вычисления BLEURT: {e}")
        return [0.0] * len(candidates)


def calculate_cosine_similarity(reference: str, candidate: str) -> float:
    """Рассчитывает косинусную близость между двумя текстами"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode([reference, candidate])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    except Exception as e:
        print(f"Ошибка вычисления косинусного сходства: {e}")
        return 0.0


def calculate_cosine_similarity_batch(texts1: list[str], texts2: list[str]) -> list[float]:
    """Рассчитывает косинусную близость между двумя наборами текстов (batch версия)"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Получаем эмбеддинги для обоих наборов текстов
        embeddings1 = model.encode(texts1)
        embeddings2 = model.encode(texts2)

        # Рассчитываем косинусную близость между соответствующими эмбеддингами
        similarities = [
            float(cosine_similarity([emb1], [emb2])[0][0])
            for emb1, emb2 in zip(embeddings1, embeddings2, strict=False)
        ]

        return similarities
    except Exception as e:
        print(f"Ошибка вычисления косинусного сходства (batch): {e}")
        return [0.0] * len(texts1)


def evaluate_batch(
    eval_data: list[dict[str, Any]],
    metrics: list[str] | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Оценивает батч данных по указанным метрикам"""
    if metrics is None:
        metrics = ["rag_triad", "bleurt", "cosine_similarity"]

    results = []

    for i, item in enumerate(eval_data):
        print(f"Оценка {i + 1}/{len(eval_data)}")

        result = {"question": item["question"], "answer": item["answer"]}

        contexts = item.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]

        expected_answer = item.get("expected_answer", "")

        # RAG Triad
        if "rag_triad" in metrics:
            rag_results = evaluate_rag_triad_single(
                item["question"], item["answer"], contexts, model_name, temperature
            )
            if "error" in rag_results:
                result["rag_triad_error"] = rag_results.get("error")
            else:
                for metric_name, payload in rag_results.items():
                    if not isinstance(payload, dict):
                        continue
                    score = payload.get("score")
                    if isinstance(score, int | float):
                        clean_name = metric_name.replace("Metric", "")
                        result[f"{clean_name}_score"] = float(score)

        # Correctness
        if "correctness" in metrics and expected_answer:
            correctness_results = evaluate_correctness_single(
                item["question"], item["answer"], expected_answer, model_name, temperature
            )
            if "error" in correctness_results:
                result["correctness_error"] = correctness_results.get("error")
            else:
                for metric_name, payload in correctness_results.items():
                    if not isinstance(payload, dict):
                        continue
                    score = payload.get("score")
                    if isinstance(score, int | float):
                        result[f"{metric_name}_score"] = float(score)

        # BLEURT
        if "bleurt" in metrics and expected_answer:
            bleurt_scores = calculate_bleurt_score([expected_answer], [item["answer"]])
            result["bleurt_score"] = bleurt_scores[0] if bleurt_scores else 0.0

        # Cosine Similarity
        if "cosine_similarity" in metrics and expected_answer:
            cosine_score = calculate_cosine_similarity(expected_answer, item["answer"])
            result["cosine_similarity"] = cosine_score

        results.append(result)

    return results


def save_evaluation_results(results: list[dict[str, Any]], output_file: str):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}

    aggregated = {}
    numeric_columns = []

    # Найдем все числовые колонки
    for result in results:
        for key, value in result.items():
            if isinstance(value, int | float) and key not in ["question", "answer"]:
                if key not in numeric_columns:
                    numeric_columns.append(key)

    # Вычислим средние значения
    for col in numeric_columns:
        values = [result[col] for result in results if isinstance(result.get(col), int | float)]
        if values:
            aggregated[f"avg_{col}"] = sum(values) / len(values)

    return aggregated
