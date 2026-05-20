from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatOpenAI(model="gpt-4-turbo", temperature=0.9)

# 1st prompt
template1=PromptTemplate(template="Write a detailed report on {topic}",
                        input_variables=["topic"] )

# 2nd prompt
template2=PromptTemplate(template="Write a 4 point summary on the following{text}",
                         input_variables=['text'])


parser=StrOutputParser()

#chain
chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"English Premier League 2023/2024"})

print(result)