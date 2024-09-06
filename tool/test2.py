from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain.pydantic_v1 import BaseModel, Field
import os

groq_key = os.environ.get("GROQ_KEY")
llm = ChatGroq(temperature=0.9, model_name="Llama3-70b-8192", groq_api_key=groq_key)

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.invoke({"a": 2, "b": 3}))
print(calculator.name)
print(calculator.description)
print(calculator.args)