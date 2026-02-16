# agent.py
import uuid
from typing import Dict, List, Optional
from memory import MemoryStore
from datetime import datetime

class Agent:
    def __init__(self, name: str, personality: str, bunker_params: dict, avatar: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.personality = personality
        self.bunker_params = bunker_params
        self.avatar = avatar
        self.mood = 0.0  # от -1 до 1
        self.relationships: Dict[str, float] = {}  # agent_id -> значение (-1..1)
        self.memory = MemoryStore()
        self.plans: List[str] = []  # текущие планы (можно позже генерировать)

        # Добавим начальное воспоминание о себе
        self.memory.add(f"Меня зовут {name}. Я {personality}. Мои параметры: {bunker_params}")

    def update_mood(self, delta: float):
        """Изменить настроение, ограничивая диапазон [-1, 1]"""
        self.mood = max(-1.0, min(1.0, self.mood + delta))

    def update_relationship(self, other_id: str, delta: float):
        """Изменить отношение к другому агенту"""
        current = self.relationships.get(other_id, 0.0)
        self.relationships[other_id] = max(-1.0, min(1.0, current + delta))