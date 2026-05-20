from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#chat template
chat_template=ChatPromptTemplate([
    ('system', "you are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

#load chat history

chat_history=[]

with open("chatbot_history.txt") as file:
    chat_history.extend(file.readlines())
    

prompt=chat_template.invoke({
    "chat_history":chat_history,
    'query': "Where is my refund?"
})

print(prompt)