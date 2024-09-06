from langchain_groq import ChatGroq
from langchain_core.tools import tool  # Import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

groq_key = os.environ.get("GROQ_KEY")
llm = ChatGroq(temperature=0.9, model_name="Llama3-70b-8192", groq_api_key=groq_key)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

class Multiply(BaseModel):
    """Multiply two numbers."""
    a: int = Field(description="first number")
    b: int = Field(description="second number")

Tools = [multiply]
tools = [Multiply]
promt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

bindllm2 = llm.bind_tools(tools)
print(bindllm2)
agent = create_tool_calling_agent(bindllm2, Tools, prompt=promt)
print(agent)
agent_executor = AgentExecutor(agent=agent, tools=Tools, verbose=True)
print(agent_executor)

query = "What is 2 multiplied by 3?"
input = {"input": query}
result = agent_executor.invoke(input)
print(result)