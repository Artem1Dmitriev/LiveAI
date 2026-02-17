import logging
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model_name, api_key=None, retries=3, base_delay=1):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            load_dotenv()
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.retries = retries
        self.base_delay = base_delay

    async def generate(self, prompt: str, system_message: str = "") -> str:
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        loop = asyncio.get_event_loop()

        for attempt in range(1, self.retries + 1):
            try:
                response = await loop.run_in_executor(None, self.model.generate_content, full_prompt)
                result = response.text
                logger.info(f"LLM response (attempt {attempt}): {result[:100]}...")
                return result
            except Exception as e:
                if attempt == self.retries or "429" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                    raise
                delay = self.base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
