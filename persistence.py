import json
import logging
from agent import Agent
from typing import Dict

logger = logging.getLogger(__name__)

def save_agents(agents: Dict[str, Agent], filepath: str):
    """Сохранить всех агентов в JSON-файл."""
    data = {aid: agent.to_dict() for aid, agent in agents.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(agents)} agents to {filepath}")

def load_agents(filepath: str) -> Dict[str, Agent]:
    """Загрузить агентов из JSON-файла. Если файл не найден, вернуть пустой словарь."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"File {filepath} not found, starting with empty agents")
        return {}
    except Exception as e:
        logger.error(f"Error loading agents: {e}")
        return {}

    agents = {}
    for aid, agent_data in data.items():
        try:
            agent = Agent.from_dict(agent_data)
            agents[aid] = agent
        except Exception as e:
            logger.error(f"Failed to load agent {aid}: {e}")
    logger.info(f"Loaded {len(agents)} agents from {filepath}")
    return agents

def save_history(history: list, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(filepath: str) -> list:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []