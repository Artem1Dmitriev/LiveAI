import os
from dotenv import load_dotenv

load_dotenv()

api_keys_str = os.getenv("GEMINI_API_KEYS", "")
if api_keys_str:
    API_KEYS = [key.strip() for key in api_keys_str.split(",") if key.strip()]
else:
    API_KEYS = [os.getenv("GEMINI_API_KEY")]

HIGH_MODELS = [
    "models/gemini-3-pro-preview",
    "models/gemini-2.5-pro",
    "models/gemini-3-pro-image-preview",
    "models/gemini-2.5-pro-preview-tts",
    "models/gemini-exp-1206",
    "models/deep-research-pro-preview-12-2025",
    "models/gemini-pro-latest",
    "models/gemma-3-27b-it",
    "models/gemini-2.5-flash-preview-09-2025",
    "models/gemini-2.5-flash-lite-preview-09-2025",
]

MEDIUM_MODELS = [
    "models/gemini-3-flash-preview",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash-preview-tts",
    "models/gemini-flash-latest",
    "models/gemini-flash-lite-latest",
    "models/gemma-3-12b-it",
    "models/gemma-3n-e4b-it",
    "models/gemma-3n-e2b-it",
    "models/gemini-2.5-computer-use-preview-10-2025",
    "models/gemini-robotics-er-1.5-preview",
]

LOW_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite",
    "models/gemma-3-4b-it",
    "models/gemma-3-1b-it",
    "models/nano-banana-pro-preview",
]

TASK_MODELS = {
    "response": HIGH_MODELS + MEDIUM_MODELS + LOW_MODELS,
    "plan": MEDIUM_MODELS + HIGH_MODELS + LOW_MODELS,
    "vote": MEDIUM_MODELS + HIGH_MODELS + LOW_MODELS,
    "sentiment": MEDIUM_MODELS + HIGH_MODELS + LOW_MODELS,
    "summarize": MEDIUM_MODELS + HIGH_MODELS + LOW_MODELS,
}

AGENTS_FILE = "agents_state.json"
HISTORY_FILE = "voting_history.json"
MEMORY_THRESHOLD = 3
BATCH_SIZE = 10
SEMAPHORE = 10