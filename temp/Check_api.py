import google.generativeai as genai
genai.configure(api_key="AIzaSyCPOdPCpyozvi1CybV_tC4mFbKdAlmg4CI")
model = genai.GenerativeModel('models/gemini-2.5-flash')
response = model.generate_content("Привет, как дела?")
print(response.text)

# for m in genai.list_models():
#     print(m.name)