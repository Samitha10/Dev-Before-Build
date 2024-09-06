
from langchain_groq import ChatGroq
from langchain_core.tools import tool  
from langchain.pydantic_v1 import BaseModel, Field
import os
from langchain.agents import tool

groq_key = os.environ.get("GROQ_KEY")
llm = ChatGroq(temperature=0.9, model_name="Llama3-70b-8192", groq_api_key=groq_key)

@tool
def multiply(x: int, y: int) -> int:
    '''Multiply two numbers together'''
    return x * y

@tool
def add(x: int, y: int) -> int:
    '''Add two numbers together'''
    return x + y

class Multiply(BaseModel):
    '''Multiply two numbers together'''
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class Add(BaseModel):
    '''Add two numbers together'''
    a: int = Field(description="first number")
    b: int = Field(description="second number")


Tools1 = [multiply, add]
bindLlm1 = llm.bind_tools(Tools1)

Tools2 = [Multiply, Add]
bindLlm2 = llm.bind_tools(Tools2)

query = "What is 2 multiplied by 3 and 2 adds 5?"
result1 = bindLlm1.invoke(query)
print(result1.tool_calls)

result2 = bindLlm2.invoke(query)
print(result2.tool_calls)

