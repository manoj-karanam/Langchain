from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()   

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(Review)

promt="""This hardware is great, but the software feels kind of bloated.
So may boilerplate apps. and my phone keeps hangin when i play PUBG"""

new_prompt="generate sentiment and summary of the review given. the review is '{promt}'"


result=structured_model.invoke(new_prompt)

print(result)