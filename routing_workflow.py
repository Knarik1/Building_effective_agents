from langchain_ollama.chat_models import ChatOllama
from typing import TypedDict, Literal
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel


# llm
LLM = ChatOllama(model="qwen2.5:1.5b-instruct")

class Router(BaseModel):
    path: Literal["story", "poem", "joke"]

LLM_with_Output = LLM.with_structured_output(Router)

class State(TypedDict):
    topic: str 
    path: str
    output: str

# nodes
def router(state: State):
    result = LLM_with_Output.invoke([
        SystemMessage("You are a helpful assistant"),
        HumanMessage(f"Decide what to genarate dependent on topic {state['topic']}")
    ])

    return {"path": result.path}

def generate_joke(state: State):
    result = LLM.invoke(state["topic"])

    return {"output": result.content}

def generate_story(state: State):
    result = LLM.invoke(state["topic"])
    
    return {"output": result.content}

def generate_poem(state: State):
    result = LLM.invoke(state["topic"])
    
    return {"output": result.content}    

def decision_maker(state: State):
    if state["path"] == "story":
        return "llm_call_1"
    elif state["path"] == "poem":
        return "llm_call_2"
    elif state["path"] == "joke":
        return "llm_call_3"
    else: 
        return 


def main():
    workflow = StateGraph(State)
    workflow.add_node("router", router)
    workflow.add_node("llm_call_1", generate_story)
    workflow.add_node("llm_call_2", generate_poem)
    workflow.add_node("llm_call_3", generate_joke)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", decision_maker, {"llm_call_1": "llm_call_1", "llm_call_2": "llm_call_2", "llm_call_3": "llm_call_3"})
    workflow.add_edge("llm_call_1", END)
    workflow.add_edge("llm_call_2", END)
    workflow.add_edge("llm_call_3", END)

    chain = workflow.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/router_workflow.png", "wb") as f:
        f.write(graph_png)

    response = chain.invoke({"topic": "Chinese medicine"})
    print(response)

if __name__ == "__main__":
    main()    