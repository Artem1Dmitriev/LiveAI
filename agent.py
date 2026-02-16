# agent.py
import uuid
from typing import Dict, List, Optional, Any
from memory import MemoryStore
from datetime import datetime

from temp.utils import POSITIVE_WORDS, NEGATIVE_WORDS


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

    async def generate_response(self,
                                message: str,
                                from_agent: Optional[str],
                                context_messages: List[Dict[str, str]],
                                game_state: Dict[str, Any],
                                llm_client) -> str:
        """
        Генерирует ответ агента. Если message пустое, агент высказывается по ситуации.
        """
        # Если есть входящее сообщение, сохраняем его
        if message:
            text_lower = message.lower()
            pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
            neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
            tone_delta = (pos_count - neg_count) * 0.1  # например, каждое слово даёт ±0.1
            # Обновляем настроение
            self.update_mood(tone_delta)
            if from_agent:
                self.memory.add(f"{from_agent} сказал: {message}")
                self.update_relationship(from_agent, tone_delta)
            else:
                self.memory.add(f"Наблюдатель сказал: {message}")

        # История диалога
        dialogue_history = ""
        if context_messages:
            for msg in context_messages[-5:]:
                sender = msg.get("from", "Unknown")
                text = msg.get("text", "")
                dialogue_history += f"{sender}: {text}\n"

        # Поиск воспоминаний, релевантных текущей ситуации
        query = message if message else "текущая ситуация в бункере, обсуждение, кто должен остаться"
        memories = self.memory.search(query, k=3)
        memories_text = "\n".join([f"- {mem}" for mem in memories])

        # Формируем промпт в зависимости от наличия входящего сообщения
        if message:
            prompt = f"""
    Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.

    Недавние воспоминания:
    {memories_text if memories_text else "Нет важных воспоминаний."}

    Ситуация в игре: {game_state}

    История последних сообщений:
    {dialogue_history}

    Тебе пришло сообщение{'' if from_agent else ' от наблюдателя'}:
    "{message}"

    Ответь на это сообщение естественно, от первого лица, кратко (1-2 предложения).
    """
        else:
            prompt = f"""
    Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.

    Недавние воспоминания:
    {memories_text if memories_text else "Нет важных воспоминаний."}

    Ситуация в игре: {game_state}
    История последних сообщений:
    {dialogue_history}

    Сейчас твоя очередь высказаться в обсуждении. Что ты скажешь? Учитывай свою личность, настрой и ситуацию. Говори кратко, как в чате (1-2 предложения).
    """

        response = await llm_client.generate(prompt)  # Убрали system_prompt

        self.memory.add(f"Я сказал: {response}")

        # Здесь позже будем обновлять настроение и отношения

        return response