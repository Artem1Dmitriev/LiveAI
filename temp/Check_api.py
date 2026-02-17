import google.generativeai as genai

genai.configure(api_key="AIzaSyCPOdPCpyozvi1CybV_tC4mFbKdAlmg4CI")

print("Модели, доступные для работы с текстом:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        # Исключаем модели генерации изображений, если они затесались
        if 'image' not in m.name or 'preview' in m.name:
            print(f"-> {m.name}")