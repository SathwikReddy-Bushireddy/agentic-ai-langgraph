from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal
from langchain_core.messages import HumanMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

class  ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

load_dotenv()
llm=ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

def chat_node(state:ChatState):
    # take query from user
    messages=state['messages']
    # send to llm
    response=llm.invoke(messages)
    # store response in state
    return {'messages':[response]}

checkpointer=MemorySaver()
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot=graph.compile(checkpointer=checkpointer)

