from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
propmt="What us the capital city of USA?"

result = model.invoke(propmt)

print(result.content)
