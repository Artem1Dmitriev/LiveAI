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