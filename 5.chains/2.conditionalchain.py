from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model="gpt-4-turbo")
parser=StrOutputParser()

class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="sentiment of the feedback provided")

parser2=PydanticOutputParser(pydantic_object=FeedBack)

prompt1=PromptTemplate(
    template="Classify the sentiment of the following review as positive or negative: {feedback} and provide a response in the following format {response_format}",
    input_variables=["feedback"],
    partial_variables={"response_format": parser2.get_format_instructions()}    
)

classifier_chain=prompt1 | model | parser2  

prompt2=PromptTemplate(template="write an appropriate response to this positive feedback: {feedback}",
                        input_variables=["feedback"])

prompt3=PromptTemplate(template="write an appropriate response to this negative feedback: {feedback}",
                        input_variables=["feedback"])


branch_chain=RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "invalid sentiment")
)

chain=classifier_chain | branch_chain

result= chain.invoke({"feedback":"The product quality is really good and I am satisfied with the purchase."})

print(result)