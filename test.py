from json import tool

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file


# load api key from .env file
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=api_key)

# tool

@tool
def duckduck(query: str):
    return DuckDuckGoSearchResults().run(query)

query = "what is today news about PTI long March"

bound_llm = llm.bind_tools([duckduck])

response = bound_llm.invoke(query)

print(response.content)