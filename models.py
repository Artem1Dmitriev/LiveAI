# models.py
from pydantic import BaseModel
from typing import Optional, Dict, List, Any


class AgentCreate(BaseModel):
    name: str
    personality: str
    bunker_params: Dict
    avatar: Optional[str] = ""

class AgentResponse(BaseModel):
    id: str
    name: str
    mood: float
    avatar: str

class AgentDetailResponse(AgentResponse):
    personality: str
    bunker_params: Dict
    relationships: Dict[str, float]
    recent_memories: List[str]
    plans: List[str]

class GameContext(BaseModel):
    recent_messages: List[Dict[str, str]]  # список сообщений: [{"from": "...", "text": "..."}]
    game_state: Dict[str, Any]              # произвольные данные о состоянии игры
    recent_events: List[str] = []

class StepRequest(BaseModel):
    context: GameContext

class StepResponse(BaseModel):
    new_messages: List[Dict[str, str]]      # [{"agent_id": "...", "text": "..."}]
    mood_updates: Dict[str, float]          # {agent_id: новое настроение}
    relationship_updates: Dict[str, float]  # пока пусто

class MessageToAgentRequest(BaseModel):
    from_agent: Optional[str] = None          # ID отправителя, если None — наблюдатель
    text: str
    context: GameContext                       # последние сообщения и состояние игры

class VoteRequest(BaseModel):
    context: GameContext  # те же последние сообщения и состояние игры

class VoteResponse(BaseModel):
    candidate_id: str
    explanation: Optional[str] = None  # для отладки/логов

class VoteResultRequest(BaseModel):
    round: int
    votes: Dict[str, str]  # voter_id -> candidate_id
    excluded_id: str       # кто был исключён в этом раунде
    alive_agents: List[str]  # ID агентов, которые остались после исключения

class EventRequest(BaseModel):
    description: str                # текст события (например, "Найден клад с едой")
    affect_mood: bool = False       # флаг, нужно ли сразу повлиять на настроение (пока не реализовано)

class RelationshipNode(BaseModel):
    id: str
    name: str
    mood: float
    avatar: str

class RelationshipEdge(BaseModel):
    source: str
    target: str
    value: float   # от -1 до 1

class RelationshipGraphResponse(BaseModel):
    nodes: List[RelationshipNode]
    edges: List[RelationshipEdge]