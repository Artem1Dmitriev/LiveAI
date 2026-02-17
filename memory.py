import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Dict, Any

class MemoryStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.memories = []  # список словарей: {'text': str, 'embedding': np.array, 'timestamp': datetime}

    def add(self, text: str):
        """Добавить воспоминание с эмбеддингом"""
        emb = self.model.encode(text, convert_to_numpy=True)
        self.memories.append({
            'text': text,
            'embedding': emb,
            'timestamp': datetime.now()
        })

    def search(self, query: str, k: int = 5) -> List[str]:
        """Поиск k самых похожих воспоминаний по косинусному сходству"""
        if not self.memories:
            return []
        query_emb = self.model.encode(query, convert_to_numpy=True)
        # нормализуем эмбеддинги для косинусного сходства (если ещё не нормализованы)
        # all-MiniLM-L6-v2 выдаёт уже нормализованные? Проверим.
        # Для безопасности вычислим косинусное сходство через скалярное произведение нормализованных векторов
        similarities = []
        for mem in self.memories:
            sim = np.dot(query_emb, mem['embedding']) / (np.linalg.norm(query_emb) * np.linalg.norm(mem['embedding']) + 1e-9)
            similarities.append(sim)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.memories[i]['text'] for i in top_indices]

    def get_recent(self, n: int = 10) -> List[str]:
        """Последние n воспоминаний (по времени)"""
        recent = sorted(self.memories, key=lambda x: x['timestamp'], reverse=True)[:n]
        return [m['text'] for m in recent]

    def to_dict(self):
        """Сериализация в словарь (без эмбеддингов)"""
        return {
            'memories': [
                {
                    'text': m['text'],
                    'timestamp': m['timestamp'].isoformat()
                }
                for m in self.memories
            ]
        }

    @classmethod
    def from_dict(cls, data):
        """Восстановление из словаря (пересчёт эмбеддингов)"""
        store = cls()
        for mem in data.get('memories', []):
            text = mem['text']
            timestamp = datetime.fromisoformat(mem['timestamp'])
            # Добавляем с эмбеддингом
            emb = store.model.encode(text, convert_to_numpy=True)
            store.memories.append({
                'text': text,
                'embedding': emb,
                'timestamp': timestamp
            })
        return store

    async def summarize_old(self, model_manager, threshold=20, batch_size=10):
        """
        Суммаризирует самые старые воспоминания, если их количество превышает threshold.
        Возвращает количество суммаризированных записей.
        """
        if len(self.memories) < threshold:
            return 0

        # Сортируем по времени, получаем индексы
        indexed = list(enumerate(self.memories))
        indexed.sort(key=lambda x: x[1]['timestamp'])
        indices_to_remove = [idx for idx, _ in indexed[:batch_size]]

        # Собираем тексты для суммаризации
        texts = [self.memories[idx]['text'] for idx in indices_to_remove]
        prompt = "Суммируй следующие воспоминания в одно короткое предложение, сохранив ключевые детали:\n" + "\n".join(
            texts)

        # Вызываем LLM
        response = await model_manager.generate_with_fallback("summarize", prompt)

        # Добавляем суммаризованное воспоминание
        self.add(f"Суммаризация: {response}")

        # Удаляем старые (в обратном порядке, чтобы не сбить индексы)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.memories[idx]

        return len(indices_to_remove)