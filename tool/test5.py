
from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool  # Import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,)

from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage

groq_key = os.environ.get("GROQ_KEY")
llm = ChatGroq(temperature=0.9, model_name="Llama3-70b-8192", groq_api_key=groq_key)


class Extractor(BaseModel):
    '''Extract the following entities'''
    category: str = Field(description="category of the product.")
    gender: str = Field(description="category of the product.")
    price: int = Field(description="price of the product.")


Tools = [Extractor]
bindLlm = llm.bind_tools(Tools)
print(bindLlm.invoke("I want some skin care products for my girl for $25").tool_calls)


memory_of_entity = ConversationBufferMemory(memory_key="history", return_messages=True)
def entity_extractor(user_message: str):
    try:
        system_message = '''
                        **You are a entity extraction assistant from unstructured user messages.**

                        **Goal:** Extract and store user preferences for product searches, including:

                        * `product_category`: string (mandatory)
                        * `gender`: string (optional)
                        * `price`: integer (optional)
                        * `product_description`: string (optional, based on user-provided details)

                        **Memory Management:**

                        * Maintain a user profile to store past preferences (`product_category`, `gender`, `price`).
                        * Update the user profile dynamically based on new information in conversations.

                        **Flagging Mechanism:**

                        * Use `flag_1` to indicate missing information the user hasn't provided yet.
                        * Use `flag_2` to indicate information the user explicitly doesn't want to specify, entities like price.
                        * Update flags based on user responses (e.g., rejections or lack of interest).

                        **Context Switching:**

                        * If the user switches to a different use case, reset information to defaults.

                        **Output:**

                        * Respond with JSON-formatted information containing extracted and stored preferences, including flags.
                        * Provide only relevant information, without additional content.

                        **Example:**

                        **User:** My sister has her 25th birthday. She care about her skin.

                        **Assistant Response:** {"product_category": "skin care", "gender": "female", "price": "flag_1", "product_description": "flag_2"}

                        **Explanation:**

                        * Assistant extracts "skin care" for `product_category` and female for `gender`.
                        * 
                        *`price` are missing (`flag_1`). 
                        * User hasn't provided a product description (`flag_2`).
                        * Do not provide additional information in your answer other than JSON output
        '''
        human_message = user_message

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),  # The persistent system prompt
                MessagesPlaceholder(variable_name="history"),  # The conversation history
                HumanMessagePromptTemplate.from_template("{input}"),  # The user's current input
            ]
        )

        # Create the conversation chain
        chain = ConversationChain(
            memory=memory_of_entity,
            llm=llm,
            verbose=False,
            prompt=prompt,

        )

        # Predict the answer
        answer = chain.predict(input=human_message)
        # Save the context
        memory_of_entity.save_context({"input": human_message}, {"output": answer})
        print(f'Assistant Response: {answer}')  


    except Exception as e:
        print(f"An error occurred: {e}")
        return e


result1 = entity_extractor("I want some skin care products for my girl for $25")
print(result1)