# agent.py
import uuid
from typing import Dict, List, Optional, Any

from memory import MemoryStore
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

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
        self._summarizing = False

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
                                model_manager) -> str:
        """
        Генерирует ответ агента. Если message пустое, агент высказывается по ситуации.
        """
        alive_agent_ids = game_state.get("alive_agents", [])
        agent_names_map = game_state.get("agent_names", {})
        alive_names = [agent_names_map.get(aid, aid) for aid in alive_agent_ids]
        game_state_desc = f"Раунд: {game_state.get('round', '?')}, живые: {', '.join(alive_names)}"

        if message:
            tone_delta = await model_manager.analyze_sentiment(message)
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

        current_plan = self.plans[-1] if self.plans else "Нет конкретного плана."

        # Формируем промпт в зависимости от наличия входящего сообщения
        if message:
            prompt = f"""
    Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.

    Недавние воспоминания:
    {memories_text if memories_text else "Нет важных воспоминаний."}

    Ситуация в игре: {game_state_desc}

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
    
    Ситуация в игре: {game_state_desc}
    Твой текущий план: {current_plan}
    История последних сообщений:
    {dialogue_history}

    Сейчас твоя очередь высказаться в обсуждении. Что ты скажешь? Учитывай свою личность, настрой, планы и ситуацию. Говори кратко, как в чате (1-2 предложения).
    """

        response = await model_manager.generate_with_fallback("response", prompt)

        self.memory.add(f"Я сказал: {response}")
        # Здесь позже будем обновлять настроение и отношения
        return response

    async def decide_vote(self, context_messages: List[Dict[str, str]], game_state: Dict[str, Any], model_manager) -> str:
        """
        Возвращает ID агента, за которого голосует этот агент.
        """
        # Собираем список живых агентов (кроме себя)
        alive_agents = game_state.get("alive_agents", [])
        others = [aid for aid in alive_agents if aid != self.id]
        if not others:
            # Если не с кем голосовать (например, остался один)
            return None

        agent_names = game_state.get("agent_names", {})
        other_names = [agent_names.get(aid, aid) for aid in others]
        current_plan = self.plans[-1] if self.plans else "Нет конкретного плана."
        # Формируем промпт
        prompt = f"""
    Ты — {self.name}. Твоя личность: {self.personality}.
    Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.
    Твой текущий план: {current_plan}


    Вы прошли обсуждение. Вот последние сообщения:
    {self._format_messages(context_messages)}
    
    Тебе нужно проголосовать за исключение одного из следующих игроков: {', '.join(other_names)}.
    Кого ты выбираешь и почему? Учитывай свою личность, план, параметры, отношения и ход обсуждения.
    Ответ дай строго в формате и больше ничего: Имя игрока.
    
    """
        response = await model_manager.generate_with_fallback("vote", prompt)
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

    def _format_messages(self, messages: List[Dict[str, str]], max_count: int = 5) -> str:
        """Форматирует список сообщений в строку для промпта."""
        if not messages:
            return ""
        result = ""
        for msg in messages[-max_count:]:
            sender = msg.get("from", "Unknown")
            text = msg.get("text", "")
            result += f"{sender}: {text}\n"
        return result

    async def update_plan(self, context_messages: List[Dict[str, str]], game_state: Dict[str, Any], model_manager, recent_events: List[str] = None) -> str:
        """
        Генерирует новый план (цель) агента на основе текущей ситуации.
        Возвращает текст плана и сохраняет его в self.plans.
        """
        events_str = "\n".join(recent_events) if recent_events else "Нет значимых событий."

        # Добавим отношения в промпт для большей релевантности
        relations_str = ", ".join(
            [f"{aid}: {val}" for aid, val in self.relationships.items()]) if self.relationships else "нейтральные"
        prompt = f"""
                Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
                Настроение: {self.mood:.2f}. Отношения с другими: {relations_str}

                Текущая ситуация в игре: {game_state}
                Последние сообщения:
                {self._format_messages(context_messages)}

                Последние события в бункере:
                {events_str}

                На основе всей этой информации сформулируй свою текущую цель (план) в игре "Бункер". Чего ты хочешь добиться в следующем раунде? Учитывай своё настроение, отношения, параметры, последние события и ход обсуждения.
                План должен быть конкретным и отражать твои намерения, например:
                - "Убедить всех, что я полезен, подчеркнув свою профессию врача."
                - "Проголосовать против Боба, потому что он кашляет и может быть опасен."
                - "Попытаться объединиться с инженером для ремонта систем."
                - "Использовать найденную еду, чтобы улучшить своё положение."
                - "Предложить план по распределению ресурсов."
                - "Защищать себя от подозрений, указывая на свои положительные качества."
                Ответ дай одной короткой фразой (1 предложение). Не используй общие фразы, будь конкретен.
                """
        response = await model_manager.generate_with_fallback("plan", prompt)
        self.plans.append(response)  # можно хранить все планы, но для демо достаточно последнего
        # Ограничим размер списка планов, чтобы не рос бесконечно
        if len(self.plans) > 10:
            self.plans = self.plans[-10:]
        return response

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'personality': self.personality,
            'bunker_params': self.bunker_params,
            'avatar': self.avatar,
            'mood': self.mood,
            'relationships': self.relationships,
            'plans': self.plans,
            'memory': self.memory.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        agent = cls(
            name=data['name'],
            personality=data['personality'],
            bunker_params=data['bunker_params'],
            avatar=data.get('avatar', '')
        )
        agent.id = data['id']
        agent.mood = data['mood']
        agent.relationships = data['relationships']
        agent.plans = data.get('plans', [])
        agent.memory = MemoryStore.from_dict(data['memory'])
        return agent

    async def summarize_memory(self, model_manager, threshold=20, batch_size=10):
        """Вызывает суммаризацию памяти агента."""
        return await self.memory.summarize_old(model_manager, threshold, batch_size)

    async def summarize_if_needed(self, model_manager, threshold=20, batch_size=10):
        """Запускает суммаризацию, если превышен порог и нет активной задачи."""
        if self._summarizing:
            logger.debug(f"Agent {self.name} already summarizing, skipping")
            return 0
        if len(self.memory.memories) < threshold:
            return 0
        self._summarizing = True
        try:
            count = await self.memory.summarize_old(model_manager, threshold, batch_size)
            logger.info(f"Agent {self.name} summarized {count} memories")
            return count
        finally:
            self._summarizing = False