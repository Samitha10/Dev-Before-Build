
from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool  # Import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

groq_key = os.environ.get("GROQ_KEY")
llm = ChatGroq(temperature=0.9, model_name="Llama3-70b-8192", groq_api_key=groq_key)



def multiply(x: int, y: int) -> int:
    '''Multiply two numbers together'''
    return x * y


def add(x: int, y: int) -> int:
    '''Add two numbers together'''
    return x + y

class Multiply(BaseModel):
    '''Multiply two numbers together'''
    x: int = Field(description="first number")
    y: int = Field(description="second number")

class Add(BaseModel):
    '''Add two numbers together'''
    x: int = Field(description="first number")
    y: int = Field(description="second number")


Tools2 = [Multiply, Add]
bindLlm2 = llm.bind_tools(Tools2)

query = "What is 2 multiplied by 3, then 2 adds 5?"
result2 = bindLlm2.invoke(query).tool_calls
print(result2)


args_by_function = {}
for fn_call in result2:
    function_name = fn_call['name']
    if function_name not in args_by_function:
        args_by_function[function_name] = {}
    args_by_function[function_name] = {**args_by_function[function_name], **fn_call['args']}


multi = args_by_function['Multiply']
adder = args_by_function['Add']
print(multi, adder)


multiplier = StructuredTool.from_function(
    func=multiply,
    name="Multiplier",
    description="multiply numbers",
    args_schema=Multiply,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)



answer = multiplier.invoke(multi)
print(answer)