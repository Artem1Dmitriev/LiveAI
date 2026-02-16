from fastapi import FastAPI, HTTPException
from agent import Agent
import uvicorn
from typing import Dict
from models import AgentCreate, AgentResponse, AgentDetailResponse, StepResponse, StepRequest, MessageToAgentRequest, \
    VoteResponse, VoteRequest, VoteResultRequest, EventRequest, RelationshipGraphResponse, RelationshipEdge, \
    RelationshipNode
from llm_client import GeminiClient  # правильный импорт
import logging
import atexit
from persistence import save_agents, load_agents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AGENTS_FILE = "agents_state.json"

app = FastAPI(title="Agent Core")

agents = load_agents(AGENTS_FILE)
llm_client = GeminiClient()  # создаём экземпляр клиента

def auto_save():
    save_agents(agents, AGENTS_FILE)

atexit.register(auto_save)

@app.post("/agents", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate):
    agent = Agent(
        name=agent_data.name,
        personality=agent_data.personality,
        bunker_params=agent_data.bunker_params,
        avatar=agent_data.avatar
    )
    agents[agent.id] = agent
    logger.info(f"Created agent {agent.name} with id {agent.id}")
    auto_save()
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
    auto_save()
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
    auto_save()


    return {"response": response_text}

@app.post("/agents/{agent_id}/vote", response_model=VoteResponse)
async def get_agent_vote(agent_id: str, request: VoteRequest):
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Подготовим game_state, добавив имена агентов
    game_state = request.context.game_state
    # Добавляем имена всех агентов (нужно для поиска по имени)
    agent_names = {aid: agents[aid].name for aid in game_state.get("alive_agents", []) if aid in agents}
    game_state["agent_names"] = agent_names

    candidate_id = await agent.decide_vote(
        context_messages=request.context.recent_messages,
        game_state=game_state,
        llm_client=llm_client
    )
    explanation = ""  # можно сохранить ответ LLM для логов
    return VoteResponse(candidate_id=candidate_id, explanation=explanation)

@app.post("/vote")
async def process_vote_results(request: VoteResultRequest):
    for agent_id in request.alive_agents:
        agent = agents.get(agent_id)
        if agent:
            agent.process_vote_results(request.votes, request.excluded_id)
    auto_save()
    return {"status": "ok"}

@app.post("/event")
async def add_event(request: EventRequest):
    """
    Добавляет глобальное событие в память всех агентов.
    """
    updated_count = 0
    for agent in agents.values():
        agent.memory.add(f"Событие: {request.description}")
        updated_count += 1
        # Если affect_mood == True, можно будет позже добавить анализ тона события и изменение настроения
        # Например: if request.affect_mood: agent.update_mood(0.1) # заглушка
    auto_save()
    return {
        "status": "ok",
        "agents_updated": updated_count,
        "description": request.description
    }

@app.get("/relationships/graph", response_model=RelationshipGraphResponse)
async def get_relationship_graph():
    """
    Возвращает граф отношений между агентами для визуализации.
    Узлы — агенты, рёбра — значения отношений (от -1 до 1).
    """
    nodes = []
    edges = []

    # Собираем узлы
    for agent_id, agent in agents.items():
        nodes.append(RelationshipNode(
            id=agent_id,
            name=agent.name,
            mood=agent.mood,
            avatar=agent.avatar
        ))

    # Собираем рёбра (для каждой пары, где отношение не 0)
    for agent_id, agent in agents.items():
        for other_id, value in agent.relationships.items():
            # Проверяем, что другой агент существует (на случай удаления)
            if other_id in agents:
                edges.append(RelationshipEdge(
                    source=agent_id,
                    target=other_id,
                    value=value
                ))

    return RelationshipGraphResponse(nodes=nodes, edges=edges)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
