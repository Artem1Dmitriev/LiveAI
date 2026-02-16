# models.py
from pydantic import BaseModel
from typing import Optional, Dict, List

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