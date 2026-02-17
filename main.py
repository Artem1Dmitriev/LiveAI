import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import logging
import atexit
import os

from config import TASK_MODELS, API_KEYS, AGENTS_FILE, HISTORY_FILE, MEMORY_THRESHOLD, BATCH_SIZE, SEMAPHORE
from agent import Agent
from models import (
    AgentCreate, AgentResponse, AgentDetailResponse, StepResponse, StepRequest,
    MessageToAgentRequest, VoteResponse, VoteRequest, VoteResultRequest,
    EventRequest, RelationshipGraphResponse, RelationshipEdge, RelationshipNode
)
from ModelManager import ModelManager
from persistence import save_agents, load_agents, load_history, save_history

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

app = FastAPI(title="Agent Core API", description="Микросервис для управления агентами в игре 'Бункер'", version="1.0.0")

agents = load_agents(AGENTS_FILE)
voting_history = load_history(HISTORY_FILE)
model_manager = ModelManager(TASK_MODELS, API_KEYS)

def auto_save():
    save_agents(agents, AGENTS_FILE)
    save_history(voting_history, HISTORY_FILE)

atexit.register(auto_save)

# ---------- Эндпоинты ----------

@app.post("/agents", response_model=AgentResponse, summary="Создать нового агента")
async def create_agent(agent_data: AgentCreate = Body(..., examples={
    "default": {
        "summary": "Пример создания агента",
        "value": {
            "name": "Алиса",
            "personality": "добрая, но вспыльчивая",
            "bunker_params": {
                "profession": "врач",
                "health": "здоров",
                "hobby": "шахматы",
                "phobia": "пауки",
                "baggage": "аптечка"
            },
            "avatar": ""
        }
    }
})):
    """
    Создаёт агента с указанными характеристиками.
    Возвращает ID агента, имя, начальное настроение (0.0) и аватар.
    """
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

@app.get("/agents", response_model=list[AgentResponse], summary="Получить список всех агентов")
async def list_agents():
    """
    Возвращает краткую информацию обо всех существующих агентах.
    """
    return [
        AgentResponse(id=a.id, name=a.name, mood=a.mood, avatar=a.avatar)
        for a in agents.values()
    ]

@app.get("/agents/{agent_id}", response_model=AgentDetailResponse, summary="Детальная информация об агенте")
async def get_agent_detail(agent_id: str):
    """
    Возвращает полную информацию об агенте: характер, параметры, отношения, последние воспоминания, планы.
    """
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

@app.post("/step", response_model=StepResponse, summary="Выполнить шаг симуляции")
async def perform_step(request: StepRequest = Body(..., examples={
    "default": {
        "summary": "Пример запроса шага",
        "value": {
            "context": {
                "recent_messages": [],
                "game_state": {
                    "round": 1,
                    "alive_agents": ["agent_id_1", "agent_id_2"],
                    "excluded": []
                },
                "recent_events": []
            }
        }
    }
})):
    """
    Запускает один шаг симуляции: все живые агенты генерируют сообщения.
    Возвращает новые сообщения и обновлённые настроения.
    """
    alive_ids = request.context.game_state.get("alive_agents", [])
    if not alive_ids:
        return StepResponse(new_messages=[], mood_updates={}, relationship_updates={})

    semaphore = asyncio.Semaphore(SEMAPHORE)

    async def process_agent(agent_id):
        async with semaphore:
            agent = agents.get(agent_id)
            if not agent:
                return None
            chosen_card, message_text = await agent.generate_initiative(
                context_messages=request.context.recent_messages,
                game_state=request.context.game_state,
                model_manager=model_manager
            )
            return {
                "agent_id": agent_id,
                "text": message_text,
                "chosen_card": chosen_card
            }

    tasks = [process_agent(aid) for aid in alive_ids]
    results = await asyncio.gather(*tasks)
    new_messages = [r for r in results if r]
    mood_updates = {aid: agents[aid].mood for aid in alive_ids if agents.get(aid)}

    for agent in agents.values():
        asyncio.create_task(agent.summarize_if_needed(model_manager, threshold=MEMORY_THRESHOLD, batch_size=BATCH_SIZE))
        asyncio.create_task(agent.update_plan(
            context_messages=request.context.recent_messages,
            game_state=request.context.game_state,
            model_manager=model_manager,
            recent_events=request.context.recent_events
        ))

    auto_save()
    return StepResponse(
        new_messages=new_messages,
        mood_updates=mood_updates,
        relationship_updates={}
    )

@app.post("/agents/{agent_id}/step", summary="Выполнить шаг для одного агента")
async def agent_step(agent_id: str, request: StepRequest = Body(..., examples={
    "default": {
        "summary": "Пример шага для одного агента",
        "value": {
            "context": {
                "recent_messages": [],
                "game_state": {"round": 1, "alive_agents": ["agent_id_1", "agent_id_2"], "excluded": []},
                "recent_events": []
            }
        }
    }
})):
    """
    Выполняет шаг симуляции только для указанного агента.
    Возвращает его сообщение и выбранную карту.
    """
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Проверяем, жив ли агент (опционально)
    alive_ids = request.context.game_state.get("alive_agents", [])
    if agent_id not in alive_ids:
        raise HTTPException(status_code=400, detail="Agent is not alive")

    # chosen_card, message_text = await agent.generate_initiative(
    #     context_messages=request.context.recent_messages,
    #     game_state=request.context.game_state,
    #     model_manager=model_manager
    # )
    message_text = await agent.generate_initiative(
            context_messages=request.context.recent_messages,
            game_state=request.context.game_state,
            model_manager=model_manager
        )

    asyncio.create_task(agent.summarize_if_needed(model_manager, threshold=MEMORY_THRESHOLD, batch_size=BATCH_SIZE))
    asyncio.create_task(agent.update_plan(
        context_messages=request.context.recent_messages,
        game_state=request.context.game_state,
        model_manager=model_manager,
        recent_events=request.context.recent_events
    ))

    auto_save()
    return {
        "agent_id": agent_id,
        "text": message_text,
        # "chosen_card": chosen_card
    }

@app.post("/agents/{agent_id}/message", summary="Отправить сообщение агенту")
async def send_message_to_agent(agent_id: str, request: MessageToAgentRequest = Body(..., examples={
    "default": {
        "summary": "Пример отправки сообщения",
        "value": {
            "from_agent": None,
            "text": "Привет, как дела?",
            "context": {
                "recent_messages": [],
                "game_state": {"round": 1, "alive_agents": ["agent_id_1", "agent_id_2"], "excluded": []},
                "recent_events": []
            }
        }
    }
})):
    """
    Отправляет сообщение указанному агенту от наблюдателя (from_agent = null) или от другого агента.
    Возвращает ответ агента.
    """
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if request.from_agent and request.from_agent not in agents:
        raise HTTPException(status_code=400, detail="Sender agent not found")

    response_text = await agent.generate_response(
        message=request.text,
        from_agent=request.from_agent,
        context_messages=request.context.recent_messages,
        game_state=request.context.game_state,
        model_manager=model_manager
    )
    for agent in agents.values():
        asyncio.create_task(agent.summarize_if_needed(model_manager, threshold=MEMORY_THRESHOLD))
    auto_save()
    return {"response": response_text}

@app.post("/agents/{agent_id}/vote", response_model=VoteResponse, summary="Получить голос агента")
async def get_agent_vote(agent_id: str, request: VoteRequest = Body(..., examples={
    "default": {
        "summary": "Пример запроса голоса",
        "value": {
            "context": {
                "recent_messages": [{"from": "Алиса", "text": "Я врач"}],
                "game_state": {"round": 1, "alive_agents": ["agent_id_1", "agent_id_2"], "excluded": []},
                "recent_events": []
            }
        }
    }
})):
    """
    Запрашивает у агента решение, за кого он голосует в текущем раунде.
    Возвращает ID выбранного кандидата.
    """
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    game_state = request.context.game_state
    agent_names = {aid: agents[aid].name for aid in game_state.get("alive_agents", []) if aid in agents}
    game_state["agent_names"] = agent_names

    candidate_id = await agent.decide_vote(
        context_messages=request.context.recent_messages,
        game_state=game_state,
        model_manager=model_manager
    )
    return VoteResponse(candidate_id=candidate_id, explanation="")

@app.post("/vote", summary="Зафиксировать результаты голосования")
async def process_vote_results(request: VoteResultRequest = Body(..., examples={
    "default": {
        "summary": "Пример результатов голосования",
        "value": {
            "round": 1,
            "votes": {"agent_id_1": "agent_id_2", "agent_id_2": "agent_id_1"},
            "excluded_id": "agent_id_1",
            "alive_agents": ["agent_id_2"]
        }
    }
})):
    """
    Принимает результаты голосования, обновляет отношения агентов и сохраняет запись в историю.
    """
    for agent_id in request.alive_agents:
        agent = agents.get(agent_id)
        if agent:
            agent.process_vote_results(request.votes, request.excluded_id)

    voting_history.append({
        "round": request.round,
        "votes": request.votes,
        "excluded_id": request.excluded_id,
        "alive_agents": request.alive_agents,
        "timestamp": datetime.now().isoformat()
    })

    auto_save()
    save_history(voting_history, HISTORY_FILE)
    return {"status": "ok"}

@app.get("/history/votes", summary="Получить историю голосований")
async def get_voting_history():
    """
    Возвращает список всех прошедших голосований.
    """
    return voting_history

@app.post("/event", summary="Добавить глобальное событие")
async def add_event(request: EventRequest = Body(..., examples={
    "default": {
        "summary": "Пример добавления события",
        "value": {
            "description": "В бункере найден запас еды",
            "affect_mood": False
        }
    }
})):
    """
    Добавляет событие в память всех агентов. Все агенты узнают о нём и смогут учитывать при следующих шагах.
    """
    updated_count = 0
    for agent in agents.values():
        agent.memory.add(f"Событие: {request.description}")
        updated_count += 1

    for agent in agents.values():
        asyncio.create_task(agent.summarize_if_needed(model_manager, threshold=MEMORY_THRESHOLD))

    auto_save()
    return {
        "status": "ok",
        "agents_updated": updated_count,
        "description": request.description
    }

@app.get("/relationships/graph", response_model=RelationshipGraphResponse, summary="Получить граф отношений")
async def get_relationship_graph():
    """
    Возвращает данные для визуализации отношений между агентами:
    - узлы: агенты (id, имя, настроение, аватар)
    - рёбра: значения отношений (source → target)
    """
    nodes = []
    edges = []

    for agent_id, agent in agents.items():
        nodes.append(RelationshipNode(
            id=agent_id,
            name=agent.name,
            mood=agent.mood,
            avatar=agent.avatar
        ))

    for agent_id, agent in agents.items():
        for other_id, value in agent.relationships.items():
            if other_id in agents:
                edges.append(RelationshipEdge(
                    source=agent_id,
                    target=other_id,
                    value=value
                ))

    return RelationshipGraphResponse(nodes=nodes, edges=edges)

@app.delete("/reset", summary="Сбросить всё состояние")
async def reset_all():
    """
    Полностью сбрасывает состояние сервера:
    - удаляет всех агентов
    - очищает историю голосований
    - удаляет файлы сохранения
    """
    agents.clear()
    voting_history.clear()

    for file in [AGENTS_FILE, HISTORY_FILE]:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Deleted {file}")

    logger.info("Reset complete: all agents and history cleared")
    return {"status": "ok", "message": "All data reset"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)