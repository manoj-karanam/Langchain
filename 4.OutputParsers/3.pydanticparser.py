from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field   


load_dotenv()

model=ChatOpenAI(model="gpt-4-turbo")


class Person(BaseModel):
    name: str = Field(description="the name of the person")
    # gt-greater than , lt-less than
    age: int = Field(gt=18, lt=100, description="The person's age")
    city: str = Field(description="the city where the person lives")
    
parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="""
    give me the name, age and city of a fictional {place} person. Make sure the age is greater than 18 and less than 100. return the response in the following format:
        {response_format}
    """,
    input_variables=["place"],
    partial_variables={"response_format": parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({"place":"Andhra Pradesh"})
print(result)