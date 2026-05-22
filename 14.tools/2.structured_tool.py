from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a:int=Field(description="The first number to multiply", required=True)
    b:int=Field(description="The second number to multiply", required=True)
    

def multiply_fnc(a:int, b:int)->int:
    return a*b

multiply_tool=StructuredTool.from_function(
    func=multiply_fnc,
    name="multiply",
    description="Multiplies two number",
    args_schema=MultiplyInput
)

result=multiply_tool.invoke({"a":5, "b":4})
print(result)