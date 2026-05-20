from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Replaced imports: Used JsonOutputParser instead of the deprecated StructuredOutputParser

load_dotenv()

model=ChatOpenAI(model="gpt-4-turbo", temperature=0.9)

class Facts(BaseModel):
    fact1: str = Field(description="first fact about black hole")
    fact2: str = Field(description="second fact about black hole")
    fact3: str = Field(description="third fact about black hole")

parser = JsonOutputParser(pydantic_object=Facts)

template=PromptTemplate(
    template="""
    give me 3 facts about {topic}, return only valid json that follows this format \n
    {response_format}""",
    input_variables=["topic"],
    partial_variables={"response_format": parser.get_format_instructions()} 
    
)

chain=template | model | parser

result=chain.invoke({"topic":"black hole"})

print(result)
