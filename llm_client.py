# llm_client.py
import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model_name="models/gemini-2.5-flash", retries=3, base_delay=1):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.retries = retries
        self.base_delay = base_delay

    async def generate(self, prompt: str, system_message: str = "") -> str:
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        loop = asyncio.get_event_loop()

        for attempt in range(1, self.retries + 1):
            try:
                logger.info(f"LLM request (attempt {attempt}): {full_prompt[:100]}...")
                response = await loop.run_in_executor(None, self.model.generate_content, full_prompt)
                result = response.text
                logger.info(f"LLM response (attempt {attempt}): {result[:100]}...")
                return result
            except Exception as e:
                logger.error(f"LLM error (attempt {attempt}): {e}")
                if attempt == self.retries:
                    fallback = "Извините, я временно не могу ответить. Попробуйте позже."
                    logger.warning(f"Using fallback: {fallback}")
                    return fallback
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

    async def analyze_sentiment(self, text: str) -> float:
        """
        Оценивает тональность текста от -1 (негативная) до 1 (позитивная).
        При ошибке возвращает 0.0.
        """
        prompt = f"Оцени эмоциональную окраску следующего сообщения от -1 (очень негативное) до 1 (очень позитивное). Ответь только числом (одним числом с плавающей точкой).\n\nСообщение: {text}"
        try:
            response = await self.generate(prompt)
            # Парсим ответ, ищем число
            import re
            match = re.search(r"-?\d+\.?\d*", response)
            if match:
                value = float(match.group())
                # Ограничиваем диапазон
                return max(-1.0, min(1.0, value))
            else:
                logger.warning(f"Could not parse sentiment from response: {response}")
                return 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0