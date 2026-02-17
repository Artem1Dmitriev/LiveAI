from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class AgentCreate(BaseModel):
    name: str = Field(..., description="Имя агента")
    personality: str = Field(..., description="Характер агента")
    bunker_params: Dict[str, Any] = Field(..., description="Параметры из игры 'Бункер'")
    avatar: Optional[str] = Field("", description="URL или идентификатор аватара")

class AgentResponse(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор агента")
    name: str = Field(..., description="Имя агента")
    mood: float = Field(..., description="Текущее настроение от -1.0 до 1.0")
    avatar: str = Field(..., description="Аватар агента")

class AgentDetailResponse(AgentResponse):
    personality: str = Field(..., description="Характер агента")
    bunker_params: Dict[str, Any] = Field(..., description="Параметры агента")
    relationships: Dict[str, int] = Field(..., description="Отношения к другим агентам: словарь {agent_id: значение от -100 до 100}")
    recent_memories: List[str] = Field(..., description="Последние воспоминания агента")
    plans: List[str] = Field(..., description="Текущие планы агента")

class GameContext(BaseModel):
    recent_messages: List[Dict[str, str]] = Field(..., description="Последние сообщения в общем чате")
    game_state: Dict[str, Any] = Field(..., description="Состояние игры: раунд, живые агенты, исключённые и т.д.")
    recent_events: List[str] = Field([], description="Список последних событий")

class StepRequest(BaseModel):
    context: GameContext = Field(..., description="Контекст для шага симуляции")

class StepResponse(BaseModel):
    new_messages: List[Dict[str, str]] = Field(..., description="Сгенерированные сообщения")
    mood_updates: Dict[str, float] = Field(..., description="Обновлённые настроения агентов")
    relationship_updates: Dict[str, float] = Field({}, description="Обновления отношений")

class MessageToAgentRequest(BaseModel):
    from_agent: Optional[str] = Field(None, description="ID отправителя (если None – наблюдатель)")
    text: str = Field(..., description="Текст сообщения")
    context: GameContext = Field(..., description="Контекст")

class VoteRequest(BaseModel):
    context: GameContext = Field(..., description="Контекст для принятия решения о голосовании")

class VoteResponse(BaseModel):
    candidate_id: str = Field(..., description="ID агента, за которого проголосовал этот агент")
    explanation: Optional[str] = Field(None, description="Пояснение")

class VoteResultRequest(BaseModel):
    round: int = Field(..., description="Номер раунда")
    votes: Dict[str, str] = Field(..., description="Результаты голосования: {voter_id: candidate_id}")
    excluded_id: str = Field(..., description="ID исключённого агента")
    alive_agents: List[str] = Field(..., description="Список ID выживших после исключения")

class EventRequest(BaseModel):
    description: str = Field(..., description="Текст события")
    affect_mood: bool = Field(False, description="Флаг, нужно ли сразу повлиять на настроение")

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
    size: str = Field(..., description="Размер бункера")
    food_supply: str = Field(..., description="Запас еды")
    equipment: str = Field(..., description="Оборудование")

class DisasterParams(BaseModel):
    type: str = Field(..., description="Тип катастрофы")
    scale: str = Field(..., description="Масштаб")
    dangers: str = Field(..., description="Опасности")

class ThreatParams(BaseModel):
    type: str = Field(..., description="Тип угрозы")
    severity: str = Field(..., description="Уровень угрозы")
    description: str = Field(..., description="Описание угрозы")