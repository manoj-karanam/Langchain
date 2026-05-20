
# contextual chatbot using langchain and google gemini model


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [SystemMessage(content="You are a helpful assistant.")]


while True:
    user_input=input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    
    if(user_input.lower() in ["exit", "quit"]):
        print("Exiting the chatbot. Goodbye!")
        break
    result=model.invoke(chat_history)
    
    chat_history.append(AIMessage(content=result.content))
    print("Chatbot:", result.content)

print("Chat history:", chat_history)

