from langchain_ollama.chat_models import ChatOllama
from typing import TypedDict, Annotated
from operator import add
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END


# llm
LLM = ChatOllama(model="qwen2.5:1.5b-instruct")

# state
class State(TypedDict):
    name: str 
    joke: str
    poem: str
    story: str
    final: str

# nodes
def call_llm_1(state: State)-> State:
    messages = [HumanMessage(f"From the season Naruto Uzumaki, tell me one joke for the character {state['name']}")]
    response = LLM.invoke(messages)

    return  {
        "joke": response.content
    }

def call_llm_2(state: State)-> State:
    messages = [HumanMessage(f"From the season Naruto Uzumaki, tell me one poem for the character {state['name']}")]
    response = LLM.invoke(messages)

    return  {
        "poem": response.content
    }

def call_llm_3(state: State)-> State:
    messages = [HumanMessage(f"From the season Naruto Uzumaki, tell me one story for the character {state['name']}")]
    response = LLM.invoke(messages)

    return  {
        "story": response.content
    }

def aggregate(state: State)-> State:
    combined = f"Here is the complete generated info {state['story']} \n\n {state['joke']} \n\n {state['poem']}"  

    return {"final": combined}


def main():
    # workflow simple (without a graph)
    # state = get_skills({"name": "Naruto"})
    # state = get_skills(state)
    # state = get_skills(state)
    # print(state)

    # workflow with graph's states
    graph = StateGraph(State)
    graph.add_node("first_node", call_llm_1)
    graph.add_node("second_node", call_llm_2)
    graph.add_node("third_node", call_llm_3)
    graph.add_node("forth_node", aggregate) 

    graph.add_edge(START, "first_node")
    graph.add_edge(START, "second_node")
    graph.add_edge(START, "third_node")
    graph.add_edge("first_node", "forth_node")
    graph.add_edge("second_node", "forth_node")
    graph.add_edge("third_node", "forth_node")
    graph.add_edge("forth_node", END)

    chain = graph.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/paralell.png", "wb") as f:
        f.write(graph_png)

    response = chain.invoke({"name": "Rain"})
    print(response["final"])



if __name__ == "__main__":
    main()    