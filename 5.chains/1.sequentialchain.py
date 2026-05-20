from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4-turbo")

prompt1 = PromptTemplate(template="generate a detailed report on the {topic}",
                            input_variables=["topic"])


prompt2 = PromptTemplate(template="generate a 3 point summary of the following text: {text}",
                            input_variables=["text"])

#entire output will be converted to string
parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({"topic":"3I Atlas Interstellar Object"})

print(result)