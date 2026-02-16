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

    async def decide_vote(self, context_messages: List[Dict[str, str]], game_state: Dict[str, Any], llm_client) -> str:
        """
        Возвращает ID агента, за которого голосует этот агент.
        """
        # Собираем список живых агентов (кроме себя)
        alive_agents = game_state.get("alive_agents", [])
        others = [aid for aid in alive_agents if aid != self.id]
        if not others:
            # Если не с кем голосовать (например, остался один)
            return None

        # Получаем имена других агентов (нужно где-то хранить маппинг id -> name)
        # Пока будем считать, что game_state содержит имена или мы передаём отдельно.
        # Для простоты передадим список имён через game_state или сделаем отдельный аргумент.
        # Упростим: пусть game_state содержит поле "agent_names" = {id: name}
        agent_names = game_state.get("agent_names", {})
        other_names = [agent_names.get(aid, aid) for aid in others]

        # Формируем промпт
        prompt = f"""
    Ты — {self.name}. Твоя личность: {self.personality}.
    Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.

    Вы прошли обсуждение. Вот последние сообщения:
    {self._format_messages(context_messages)}

    Тебе нужно проголосовать за исключение одного из следующих игроков: {', '.join(other_names)}.
    Кого ты выбираешь и почему? Учитывай свою личность, параметры, отношения и ход обсуждения.
    Ответ дай строго в формате: Имя игрока (причина).
    """
        response = await llm_client.generate(prompt)
        # Парсим ответ: ожидаем, что первое слово — имя кандидата
        # Простейший парсинг: ищем имя из списка в ответе
        chosen_name = None
        for name in other_names:
            if name in response:
                chosen_name = name
                break
        if chosen_name is None:
            # Если не нашли, берём первого (запасной вариант)
            chosen_name = other_names[0] if other_names else None

        # Находим ID по имени (обратный словарь)
        name_to_id = {v: k for k, v in agent_names.items()}
        candidate_id = name_to_id.get(chosen_name)
        return candidate_id

    def process_vote_results(self, votes: Dict[str, str], excluded_id: str):
        """
        Обновляет отношения на основе голосования.
        """
        # Если этот агент голосовал против кого-то, ухудшаем отношение к тому кандидату
        my_vote = votes.get(self.id)
        if my_vote and my_vote != excluded_id:  # голосовал против того, кто не исключён (или исключён)
            # Ухудшаем отношение к кандидату, за которого голосовал
            self.update_relationship(my_vote, -0.2)
        # Если кто-то голосовал против этого агента
        for voter, candidate in votes.items():
            if candidate == self.id and voter != self.id:
                # Ухудшаем отношение к тому, кто голосовал против нас
                self.update_relationship(voter, -0.1)
        # Если агента исключили, он больше не участвует, но можно ничего не делать