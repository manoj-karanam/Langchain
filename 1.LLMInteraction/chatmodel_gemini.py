from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
propmt="Do you remember my name?"

result = model.invoke(propmt)

print(result.content)
