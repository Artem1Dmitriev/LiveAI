# main.py
from fastapi import FastAPI, HTTPException
from models import AgentCreate, AgentResponse, AgentDetailResponse
from agent import Agent
import uvicorn
from typing import Dict

app = FastAPI(title="Agent Core")

# Хранилище агентов в памяти
agents: Dict[str, Agent] = {}

@app.post("/agents", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate):
    """Создать нового агента"""
    agent = Agent(
        name=agent_data.name,
        personality=agent_data.personality,
        bunker_params=agent_data.bunker_params,
        avatar=agent_data.avatar
    )
    agents[agent.id] = agent
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        mood=agent.mood,
        avatar=agent.avatar
    )

@app.get("/agents", response_model=list[AgentResponse])
async def list_agents():
    """Получить список всех агентов (кратко)"""
    return [
        AgentResponse(id=a.id, name=a.name, mood=a.mood, avatar=a.avatar)
        for a in agents.values()
    ]

@app.get("/agents/{agent_id}", response_model=AgentDetailResponse)
async def get_agent_detail(agent_id: str):
    """Получить детальную информацию об агенте"""
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    recent = agent.memory.get_recent(5)  # последние 5 воспоминаний
    return AgentDetailResponse(
        id=agent.id,
        name=agent.name,
        mood=agent.mood,
        avatar=agent.avatar,
        personality=agent.personality,
        bunker_params=agent.bunker_params,
        relationships=agent.relationships,
        recent_memories=recent,
        plans=agent.plans
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)