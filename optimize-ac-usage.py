#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("chillwise_ac_usage_dataset.csv")
print(df.head())


# In[3]:


from sklearn.metrics import r2_score


# In[4]:


from xgboost import XGBRegressor

x_train = df.drop("Electricity_cost",axis=1)
y_train = df["Electricity_cost"]

xgb = XGBRegressor()

xgb.fit(x_train,y_train)


# In[5]:


from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage 
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool


# In[6]:


class State(TypedDict):
    messages:Annotated[list,add_messages]


# In[7]:


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-70b-8192")


# In[8]:


from langchain.tools import tool

@tool
def mlModel(
    Hours_per_day: int,
    Temp_set: int,
    Room_size: int,
    Outside_temp: int,
    AC_type: float
) -> str:
    """
    Predict the electricity cost based on AC usage parameters.

    Args:
        Hours_per_day (int): Number of hours AC is used per day.
        Temp_set (int): Temperature set on the AC.
        Room_size (int): Size of the room in square feet.
        Outside_temp (int): Current outside temperature.
        AC_type (float): AC tonnage (e.g., 1.0, 1.5, 2.0).

    Returns:
        str: Predicted electricity cost in ₹.
    """
    df_test = pd.DataFrame({
        "Hours_per_day": [Hours_per_day],
        "Temp_set": [Temp_set],
        "Room_size": [Room_size],
        "Outside_temp": [Outside_temp],
        "AC_type": [AC_type]
    })
    prediction = xgb.predict(df_test)
    return f"The predicted electricity cost is ₹{prediction[0]:.2f}"


# In[ ]:





# In[9]:


tools = [mlModel]
tool_calling_llm = llm.bind_tools(tools)
def chat_llm(state: State):
    result = tool_calling_llm.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}  # <-- Append new response


# In[ ]:


def decide_next(state):
    if "tool_call" in state:  # example condillm_with_tooltion
        return "tool"
    return "__end__"


# In[20]:


graph_builder = StateGraph(State)
graph_builder.add_node("llm_with_tool", chat_llm)
graph_builder.add_node("tool", ToolNode(tools))

graph_builder.add_edge(START, "llm_with_tool")



# tool always returns to llm_with_tool (never ends)
graph_builder.add_edge("llm_with_tool", "tool")

# Remove this — redundant and might allow accidental END without condition
# graph_builder.add_edge("llm_with_tool", END)

graph = graph_builder.compile()


# In[21]:


from IPython.display import Image,display
display(Image(graph.get_graph().draw_mermaid_png()))


# In[22]:


state = {
    "messages": [
        {
            "role": "user",
            "content": "I use my AC for 6 hours a day at 24 degrees. My room is 120 sq ft, the outside temperature is 35°C, and it's a 1.5 ton AC. Can you tell me the electricity cost?"
        }
    ]
}
response = graph.invoke(state)
print(response["messages"])


# In[14]:


for m in response["messages"]:
    m.pretty_print()


# In[ ]:





# In[ ]:




