# llm_client.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

load_dotenv()

class GeminiClient:
    def __init__(self, model_name="models/gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    async def generate(self, prompt: str, system_message: str = "") -> str:
        """Асинхронно генерирует ответ. system_message пока игнорируем, в Gemini можно передать как context"""
        # Gemini не имеет отдельного system prompt, можно передать как часть истории или в контекст
        # Пока просто объединим с prompt
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        # Запускаем синхронный вызов в отдельном потоке
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.model.generate_content, full_prompt)
        return response.text