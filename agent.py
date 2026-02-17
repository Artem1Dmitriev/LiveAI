import uuid
from typing import Dict, List, Optional, Any

from memory import MemoryStore
import logging

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, name: str, personality: str, bunker_params: dict, avatar: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.personality = personality
        self.bunker_params = bunker_params
        self.avatar = avatar
        self.mood = 0.0
        self.relationships: Dict[str, float] = {}
        self.memory = MemoryStore()
        self.plans: List[str] = []
        self.revealed_cards: List[str] = []

        self.memory.add(f"Меня зовут {name}. Я {personality}. Мои параметры: {bunker_params}")
        self._summarizing = False

    def update_mood(self, delta: float):
        """Изменить настроение, ограничивая диапазон [-1, 1]"""
        self.mood = max(-1.0, min(1.0, self.mood + delta))

    def update_relationship(self, other_id: str, delta: float):
        """Изменить отношение к другому агенту"""
        current = self.relationships.get(other_id, 0.0)
        self.relationships[other_id] = max(-1.0, min(1.0, current + delta))

    async def generate_initiative(self, context_messages: List[Dict[str, str]], game_state: Dict[str, Any],
                                  model_manager) -> tuple[str, str]:
        bunker = game_state.get("bunker") or (globals().get('current_bunker') if 'current_bunker' in globals() else {})
        disaster = game_state.get("disaster") or (
            globals().get('current_disaster') if 'current_disaster' in globals() else {})
        threat = game_state.get("threat") or (globals().get('current_threat') if 'current_threat' in globals() else {})

        # Формируем строки для вставки
        bunker_info = f"Размер: {bunker.get('size', 'неизвестно')}, запас еды: {bunker.get('food_supply', 'неизвестно')}, оборудование: {bunker.get('equipment', 'неизвестно')}" if bunker else "Информация о бункере отсутствует"
        disaster_info = f"Тип: {disaster.get('type', 'неизвестно')}, масштаб: {disaster.get('scale', 'неизвестно')}, опасности: {disaster.get('dangers', 'неизвестно')}" if disaster else "Информация о катастрофе отсутствует"
        threat_info = f"Тип: {threat.get('type', 'неизвестно')}, уровень: {threat.get('severity', 'неизвестно')}, описание: {threat.get('description', 'неизвестно')}" if threat else "Информация об угрозе отсутствует"

        """
        Генерирует инициативное высказывание: агент выбирает одну НЕРАСКРЫТУЮ карту и объясняет, почему она делает его ценным.
        Возвращает (название_карты, текст_высказывания).
        """
        dialogue_history = self._format_messages(context_messages)
        memories = self.memory.search("текущая ситуация в бункере, обсуждение, кто должен остаться", k=3)
        memories_text = "\n".join([f"- {mem}" for mem in memories]) if memories else "Нет важных воспоминаний."

        all_cards = ["profession", "age", "gender", "health", "hobby", "baggage", "personality"]
        available_cards = [card for card in all_cards if card not in self.revealed_cards]

        if not available_cards:
            prompt = f"""
    Ты — {self.name}. Характер: {self.personality}. Настроение: {self.mood:.2f}.

    Ты уже раскрыл все свои карты. Сейчас просто выскажись, почему ты должен остаться, ссылаясь на уже известные качества. Говори кратко одним предложением.
    """
            response = await model_manager.generate_with_fallback("response", prompt)
            chosen_card = "none"
            message_text = response.strip()
        else:
            chosen_card = available_cards[0]
            if chosen_card == "personality":
                card_value = self.personality
            else:
                card_value = self.bunker_params.get(chosen_card, "неизвестно")
            context_str = f"Бункер: {bunker_info}\nКатастрофа: {disaster_info}\nУгроза: {threat_info}"
            card_usefulness = await model_manager.evaluate_card(chosen_card, card_value, context_str)
            self.update_mood(card_usefulness * 0.3)
            # cards_info = ""
            # for card in available_cards:
            #     value = self.bunker_params.get(card, "неизвестно")
            #     cards_info += f"- {card}: {value}\n"
            prompt = f"""
                Ты — {self.name}. Характер: {self.personality}. Настроение: {self.mood:.2f}.
            
                Ранее ты уже раскрыл: {', '.join(self.revealed_cards) if self.revealed_cards else 'пока ничего'}.
                Обстановка в бункере:
                {bunker_info}
            
                Катастрофа, которая произошла:
                {disaster_info}
            
                Угроза снаружи:
                {threat_info}
                    
                Недавние воспоминания:
                {memories_text}
            
                Ситуация в игре: {game_state}
                История последних сообщений:
                {dialogue_history}
            
                Сейчас твоя очередь высказаться.
                Ты раскрываешь карту «{chosen_card}»: {card_value}. объясни, почему именно эта твоя характеристика делает тебя ценным для выживания группы. Говори только о выбранной карте, НЕ УПОМИНАЙ другие свои характеристики (здоровье, хобби, фобию, багаж, если не выбрал их). Не говори о том, что ты уже раскрывал ранее.
                Говори кратко. Одно предложение.
                Твой ответ должен содержать:
                1. Название карты, которую ты раскрываешь (например, "profession", "health", "hobby", "phobia", "baggage").
                2. Само объяснение.
            
                Формат: сначала укажи карту в квадратных скобках, например [profession], а затем напиши своё высказывание. Не используй квадратные скобки больше нигде.
                """
            response = await model_manager.generate_with_fallback("response", prompt)

            import re
            match = re.search(r'\[(.*?)\]', response)
            if match:
                chosen_card = match.group(1).strip()
                message_text = re.sub(r'\[.*?\]', '', response).strip()
                if chosen_card not in available_cards:
                    chosen_card = available_cards[0] if available_cards else "none"
            else:
                chosen_card = available_cards[0] if available_cards else "none"
                message_text = response.strip()

        if chosen_card != "none" and chosen_card not in self.revealed_cards:
            self.revealed_cards.append(chosen_card)

        self.memory.add(f"Я раскрыл карту [{chosen_card}]: {message_text}")
        logger.info(f"Agent {self.name} initiative: [{chosen_card}] {message_text}")
        return response

    async def generate_response(self,
                                message: str,
                                from_agent: Optional[str],
                                context_messages: List[Dict[str, str]],
                                game_state: Dict[str, Any],
                                model_manager) -> str:
        """
        Генерирует ответ агента. Если message пустое, агент высказывается по ситуации.
        """
        bunker = game_state.get("bunker") or (globals().get('current_bunker') if 'current_bunker' in globals() else {})
        disaster = game_state.get("disaster") or (
            globals().get('current_disaster') if 'current_disaster' in globals() else {})
        threat = game_state.get("threat") or (globals().get('current_threat') if 'current_threat' in globals() else {})

        # Формируем строки для вставки
        bunker_info = f"Размер: {bunker.get('size', 'неизвестно')}, запас еды: {bunker.get('food_supply', 'неизвестно')}, оборудование: {bunker.get('equipment', 'неизвестно')}" if bunker else "Информация о бункере отсутствует"
        disaster_info = f"Тип: {disaster.get('type', 'неизвестно')}, масштаб: {disaster.get('scale', 'неизвестно')}, опасности: {disaster.get('dangers', 'неизвестно')}" if disaster else "Информация о катастрофе отсутствует"
        threat_info = f"Тип: {threat.get('type', 'неизвестно')}, уровень: {threat.get('severity', 'неизвестно')}, описание: {threat.get('description', 'неизвестно')}" if threat else "Информация об угрозе отсутствует"

        alive_agent_ids = game_state.get("alive_agents", [])
        agent_names_map = game_state.get("agent_names", {})
        alive_names = [agent_names_map.get(aid, aid) for aid in alive_agent_ids]
        game_state_desc = f"Раунд: {game_state.get('round', '?')}, живые: {', '.join(alive_names)}"

        if message:
            tone_delta = await model_manager.analyze_sentiment(message)
            self.update_mood(tone_delta)
            if from_agent:
                self.memory.add(f"{from_agent} сказал: {message}")
                self.update_relationship(from_agent, tone_delta)
            else:
                self.memory.add(f"Наблюдатель сказал: {message}")

        dialogue_history = ""
        if context_messages:
            for msg in context_messages[-5:]:
                sender = msg.get("from", "Unknown")
                text = msg.get("text", "")
                dialogue_history += f"{sender}: {text}\n"

        query = message if message else "текущая ситуация в бункере, обсуждение, кто должен остаться"
        memories = self.memory.search(query, k=3)
        memories_text = "\n".join([f"- {mem}" for mem in memories])

        current_plan = self.plans[-1] if self.plans else "Нет конкретного плана."

        if message:
            prompt = f"""
    Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
    Настроение: {self.mood:.2f}.
    
    Обстановка в бункере:
    {bunker_info}
    
    Катастрофа, которая произошла:
    {disaster_info}
    
    Угроза снаружи:
    {threat_info}
    
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
        return response

    async def decide_vote(self, context_messages: List[Dict[str, str]], game_state: Dict[str, Any], model_manager) -> str:
        """
        Возвращает ID агента, за которого голосует этот агент.
        """
        bunker = game_state.get("bunker") or (globals().get('current_bunker') if 'current_bunker' in globals() else {})
        disaster = game_state.get("disaster") or (
            globals().get('current_disaster') if 'current_disaster' in globals() else {})
        threat = game_state.get("threat") or (globals().get('current_threat') if 'current_threat' in globals() else {})

        # Формируем строки для вставки
        bunker_info = f"Размер: {bunker.get('size', 'неизвестно')}, запас еды: {bunker.get('food_supply', 'неизвестно')}, оборудование: {bunker.get('equipment', 'неизвестно')}" if bunker else "Информация о бункере отсутствует"
        disaster_info = f"Тип: {disaster.get('type', 'неизвестно')}, масштаб: {disaster.get('scale', 'неизвестно')}, опасности: {disaster.get('dangers', 'неизвестно')}" if disaster else "Информация о катастрофе отсутствует"
        threat_info = f"Тип: {threat.get('type', 'неизвестно')}, уровень: {threat.get('severity', 'неизвестно')}, описание: {threat.get('description', 'неизвестно')}" if threat else "Информация об угрозе отсутствует"

        alive_agents = game_state.get("alive_agents", [])
        others = [aid for aid in alive_agents if aid != self.id]
        if not others:
            return None

        agent_names = game_state.get("agent_names", {})
        other_names = [agent_names.get(aid, aid) for aid in others]
        current_plan = self.plans[-1] if self.plans else "Нет конкретного плана."
        prompt = f"""
    Ты — {self.name}. Твоя личность: {self.personality}.
    Обстановка в бункере:
    {bunker_info}
    
    Катастрофа, которая произошла:
    {disaster_info}
    
    Угроза снаружи:
    {threat_info}
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
        chosen_name = None
        for name in other_names:
            if name in response:
                chosen_name = name
                break
        if chosen_name is None:
            chosen_name = other_names[0] if other_names else None

        name_to_id = {v: k for k, v in agent_names.items()}
        candidate_id = name_to_id.get(chosen_name)
        return candidate_id

    def update_mood_from_relationships(self):
        if not self.relationships:
            return
        avg_rel = sum(self.relationships.values()) / len(self.relationships)
        self.update_mood(avg_rel * 0.1)

    def process_vote_results(self, votes: Dict[str, str], excluded_id: str):
        """
        Обновляет отношения и настроение на основе голосования.
        """
        my_vote = votes.get(self.id)

        if my_vote == excluded_id:
            self.update_mood(0.2)
        elif my_vote and my_vote != excluded_id:
            self.update_mood(-0.1)

        votes_against_me = [voter for voter, candidate in votes.items() if candidate == self.id and voter != self.id]
        if votes_against_me:
            mood_delta = -0.05 * len(votes_against_me)
            self.update_mood(mood_delta)

        if excluded_id in self.relationships and self.relationships[excluded_id] > 0.5:
            self.update_mood(-0.2)
        if excluded_id in self.relationships and self.relationships[excluded_id] < -0.5:
            self.update_mood(0.2)

        if my_vote and my_vote != excluded_id:
            self.update_relationship(my_vote, -0.2)
        for voter, candidate in votes.items():
            if candidate == self.id and voter != self.id:
                self.update_relationship(voter, -0.1)
        self.update_mood_from_relationships()

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
        bunker = game_state.get("bunker") or (globals().get('current_bunker') if 'current_bunker' in globals() else {})
        disaster = game_state.get("disaster") or (
            globals().get('current_disaster') if 'current_disaster' in globals() else {})
        threat = game_state.get("threat") or (globals().get('current_threat') if 'current_threat' in globals() else {})

        # Формируем строки для вставки
        bunker_info = f"Размер: {bunker.get('size', 'неизвестно')}, запас еды: {bunker.get('food_supply', 'неизвестно')}, оборудование: {bunker.get('equipment', 'неизвестно')}" if bunker else "Информация о бункере отсутствует"
        disaster_info = f"Тип: {disaster.get('type', 'неизвестно')}, масштаб: {disaster.get('scale', 'неизвестно')}, опасности: {disaster.get('dangers', 'неизвестно')}" if disaster else "Информация о катастрофе отсутствует"
        threat_info = f"Тип: {threat.get('type', 'неизвестно')}, уровень: {threat.get('severity', 'неизвестно')}, описание: {threat.get('description', 'неизвестно')}" if threat else "Информация об угрозе отсутствует"

        events_str = "\n".join(recent_events) if recent_events else "Нет значимых событий."

        relations_str = ", ".join(
            [f"{aid}: {val}" for aid, val in self.relationships.items()]) if self.relationships else "нейтральные"
        prompt = f"""
                Ты — {self.name}. Характер: {self.personality}. Параметры: {self.bunker_params}.
                Настроение: {self.mood:.2f}. Отношения с другими: {relations_str}
                Обстановка в бункере:
                {bunker_info}
                
                Катастрофа, которая произошла:
                {disaster_info}
                
                Угроза снаружи:
                {threat_info}
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
        self.plans.append(response)
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
            'memory': self.memory.to_dict(),
            'revealed_cards': self.revealed_cards,
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
        agent.revealed_cards = data.get('revealed_cards', [])
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