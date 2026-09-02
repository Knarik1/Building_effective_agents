from langchain_ollama.chat_models import ChatOllama
from typing import TypedDict
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool

@tool
def mult(a: int, b: int):
    """ Multiplying 2 integer numbers"""
    return a * b 

@tool
def add(a: int, b: int):
    """ Adding 2 integer numbers"""
    return a + b

@tool
def divide(a: int, b: int):
    """ Dividing first number into the second one"""
    return a / b


LLM = ChatOllama(model="qwen2.5:1.5b-instruct")
tools = [mult, add, divide]
tool_by_name = {tool.name: tool for tool in tools}
LLM_with_Tools = LLM.bind_tools(tools)

class State(TypedDict):
    messages: list

# nodes
def llm_call(state: MessagesState):
    response = LLM_with_Tools.invoke([
        SystemMessage("You are a helpful assistant performing arithmetic")
    ] + state["messages"])

    return {"messages": response}

def tool_call(state: dict):
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tool_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}

    
def should_continue(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "Action"
    else: 
        return "Finish"

def main():
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_call", tool_call)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "tool_call", "Finish": END})

    chain = agent_builder.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/agentic_workflow.png", "wb") as f:
        f.write(graph_png)

    response = chain.invoke({"messages": ["Add 3 and 4 then take the output and multiply the result by 5."]})
    for m in response["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()