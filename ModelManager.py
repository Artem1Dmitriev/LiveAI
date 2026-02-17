from typing import List, Dict
from llm_client import GeminiClient
import logging
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, task_models: Dict[str, List[str]], api_keys: List[str]):
        self.task_models = task_models
        self.api_keys = api_keys
        self.current_key_index = 0
        self._clients_cache = {}

    def _get_client(self, model: str, key: str):
        cache_key = (model, key)
        if cache_key not in self._clients_cache:
            self._clients_cache[cache_key] = GeminiClient(model_name=model, api_key=key)
        return self._clients_cache[cache_key]

    async def generate_with_fallback(self, task: str, prompt: str, system_message: str = "") -> str:
        models = self.task_models.get(task, self.task_models["response"])
        for model in models:
            for key in self.api_keys:
                try:
                    client = self._get_client(model, key)
                    result = await client.generate(prompt, system_message)
                    logger.info(f"Success with model {model}...")
                    return result
                except Exception as e:
                    continue
        logger.critical(f"All model/key combinations failed for task {task}")
        return "Извините, я временно не могу ответить. Попробуйте позже."

    async def analyze_sentiment(self, text: str) -> float:
        prompt = f"Оцени эмоциональную окраску сообщения от -1 до 1. Ответь только числом (одним числом с плавающей точкой). Никаких пояснений.\nСообщение: {text}"
        try:
            response = await self.generate_with_fallback("sentiment", prompt)
            logger.info(f"Sentiment raw response: {response}")
            import re
            match = re.search(r"-?\d+\.?\d*", response)
            if match:
                value = float(match.group())
                logger.info(f"Sentiment parsed value: {value}")
                return max(-1.0, min(1.0, value))
            else:
                logger.warning(f"Could not parse sentiment from response: {response}")
                return 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0

    async def evaluate_card(self, card_name: str, card_value: str, context: str) -> float:
        """
        Оценивает, насколько полезна данная карта в текущем контексте (bunker/disaster/threat).
        Возвращает число от -1 до 1.
        """
        prompt = f"""
Оцени полезность характеристики "{card_name}" со значением "{card_value}" для выживания группы в условиях:
{context}
Твой ответ должен быть только одним числом от -1 (очень вредно) до 1 (очень полезно). Никаких пояснений.
"""
        response = await self.generate_with_fallback("sentiment", prompt)
        logger.info(f"Evaluate card: name={card_name}, value={card_value}")
        logger.info(f"Evaluate card context: {context[:200]}...")
        logger.info(f"Evaluate card raw response: {response}")
        try:
            value = float(response.strip())
            logger.info(f"Evaluate card parsed value: {value}")
            return max(-1.0, min(1.0, value))
        except Exception as e:
            logger.error(f"Failed to parse evaluate_card response: {response}, error: {e}")
            return 0.0