from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class AgentCreate(BaseModel):
    name: str = Field(..., description="Имя агента (например, 'Алиса')")
    personality: str = Field(..., description="Характер агента (например, 'добрая, но вспыльчивая')")
    bunker_params: Dict[str, Any] = Field(..., description="Параметры из игры 'Бункер': профессия, здоровье, хобби, фобия, багаж и т.д.")
    avatar: Optional[str] = Field("", description="URL или идентификатор аватара (опционально)")

class AgentResponse(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор агента")
    name: str = Field(..., description="Имя агента")
    mood: float = Field(..., description="Текущее настроение от -1.0 (плохое) до 1.0 (отличное)")
    avatar: str = Field(..., description="Аватар агента (если не задан, пустая строка)")

class AgentDetailResponse(AgentResponse):
    personality: str = Field(..., description="Характер агента")
    bunker_params: Dict[str, Any] = Field(..., description="Параметры агента")
    relationships: Dict[str, int] = Field(..., description="Отношения к другим агентам: словарь {agent_id: значение от -100 до 100}")
    recent_memories: List[str] = Field(..., description="Последние 5-10 воспоминаний агента (текст)")
    plans: List[str] = Field(..., description="Текущие планы агента (список, последний – актуальный)")

class GameContext(BaseModel):
    recent_messages: List[Dict[str, str]] = Field(..., description="Последние сообщения в общем чате. Каждый элемент: {\"from\": \"имя или ID\", \"text\": \"сообщение\"}")
    game_state: Dict[str, Any] = Field(..., description="Состояние игры: раунд, живые агенты, исключённые и т.д.")
    recent_events: List[str] = Field([], description="Список последних событий (например, ['Найден запас еды'])")

class StepRequest(BaseModel):
    context: GameContext = Field(..., description="Контекст для шага симуляции")

class StepResponse(BaseModel):
    new_messages: List[Dict[str, str]] = Field(..., description="Сгенерированные сообщения: [{\"agent_id\": \"...\", \"text\": \"...\"}]")
    mood_updates: Dict[str, float] = Field(..., description="Обновлённые настроения агентов: {agent_id: новое_настроение}")
    relationship_updates: Dict[str, float] = Field({}, description="Обновления отношений (пока пусто)")

class MessageToAgentRequest(BaseModel):
    from_agent: Optional[str] = Field(None, description="ID отправителя (если None – сообщение от наблюдателя)")
    text: str = Field(..., description="Текст сообщения")
    context: GameContext = Field(..., description="Контекст (последние сообщения и состояние игры)")

class VoteRequest(BaseModel):
    context: GameContext = Field(..., description="Контекст для принятия решения о голосовании")

class VoteResponse(BaseModel):
    candidate_id: str = Field(..., description="ID агента, за которого проголосовал этот агент")
    explanation: Optional[str] = Field(None, description="Пояснение (может быть пустым)")

class VoteResultRequest(BaseModel):
    round: int = Field(..., description="Номер раунда")
    votes: Dict[str, str] = Field(..., description="Результаты голосования: {voter_id: candidate_id}")
    excluded_id: str = Field(..., description="ID исключённого агента")
    alive_agents: List[str] = Field(..., description="Список ID выживших после исключения")

class EventRequest(BaseModel):
    description: str = Field(..., description="Текст события (например, 'В бункере найден запас еды')")
    affect_mood: bool = Field(False, description="Флаг, нужно ли сразу повлиять на настроение агентов (пока не реализовано)")

class RelationshipNode(BaseModel):
    id: str = Field(..., description="ID агента")
    name: str = Field(..., description="Имя агента")
    mood: float = Field(..., description="Настроение агента")
    avatar: str = Field(..., description="Аватар агента")

class RelationshipEdge(BaseModel):
    from_: str = Field(..., alias="from", description="ID исходного агента")
    to: str = Field(..., description="ID целевого агента")
    value: int = Field(..., description="Значение отношения от -100 до 100")

class RelationshipGraphResponse(BaseModel):
    nodes: List[RelationshipNode] = Field(..., description="Список узлов (агентов)")
    edges: List[RelationshipEdge] = Field(..., description="Список рёбер (отношений)")

class BunkerParams(BaseModel):
    size: str = Field(..., description="Размер бункера (например, 'маленький', 'средний', 'большой')")
    food_supply: str = Field(..., description="Запас еды (например, 'неделя', 'месяц', 'год')")
    equipment: str = Field(..., description="Оборудование (например, 'медицинское', 'техническое', 'отсутствует')")

class DisasterParams(BaseModel):
    type: str = Field(..., description="Тип катастрофы (например, 'ядерная война', 'пандемия', 'наводнение')")
    scale: str = Field(..., description="Масштаб (критический, континентальный, планетарный, местный и т.п.)")
    dangers: str = Field(..., description="Опасности (например, 'ядерная зима 50 лет', 'заражено 90% населения')")

class ThreatParams(BaseModel):
    type: str = Field(..., description="Тип угрозы (например, 'мутанты', 'бандиты', 'радиация')")
    severity: str = Field(..., description="Уровень угрозы (например, 'низкий', 'средний', 'критический')")
    description: str = Field(..., description="Описание угрозы")