from fastapi import FastAPI, HTTPException
from agent import Agent
import uvicorn
from typing import Dict
from models import AgentCreate, AgentResponse, AgentDetailResponse, StepResponse, StepRequest, MessageToAgentRequest
from llm_client import GeminiClient  # правильный импорт

app = FastAPI(title="Agent Core")

agents: Dict[str, Agent] = {}
llm_client = GeminiClient()  # создаём экземпляр клиента

@app.post("/agents", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate):
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
    return [
        AgentResponse(id=a.id, name=a.name, mood=a.mood, avatar=a.avatar)
        for a in agents.values()
    ]

@app.get("/agents/{agent_id}", response_model=AgentDetailResponse)
async def get_agent_detail(agent_id: str):
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    recent = agent.memory.get_recent(5)
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

@app.post("/step", response_model=StepResponse)
async def perform_step(request: StepRequest):
    alive_ids = request.context.game_state.get("alive_agents", [])
    if not alive_ids:
        return StepResponse(new_messages=[], mood_updates={}, relationship_updates={})

    new_messages = []
    mood_updates = {}

    for agent_id in alive_ids:
        agent = agents.get(agent_id)
        if not agent:
            continue

        response_text = await agent.generate_response(
            message="",
            from_agent=None,
            context_messages=request.context.recent_messages,
            game_state=request.context.game_state,
            llm_client=llm_client  # передаём экземпляр клиента
        )

        new_messages.append({
            "agent_id": agent_id,
            "text": response_text
        })
        mood_updates[agent_id] = agent.mood

    return StepResponse(
        new_messages=new_messages,
        mood_updates=mood_updates,
        relationship_updates={}
    )

@app.post("/agents/{agent_id}/message")
async def send_message_to_agent(agent_id: str, request: MessageToAgentRequest):
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    response_text = await agent.generate_response(
        message=request.text,
        from_agent=request.from_agent,
        context_messages=request.context.recent_messages,
        game_state=request.context.game_state,
        llm_client=llm_client
    )

    return {"response": response_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)