from typing import TypedDict
from langchain_ollama.chat_models import ChatOllama
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, END, START

LLM = ChatOllama(model="qwen2.5:1.5b-instruct")

# State
class State(TypedDict):
    topic: str
    story: str 
    updated_story: str 
    final_story: str

# Nodes
def generate_story(state: State)-> State:
    messages = [HumanMessage(f"Generate a story given this topic {state['topic']}, make it laconic, up to 5 sentences")]
    response = LLM.invoke(messages)
    state["story"] = response.content

    return state

def revise_story(state: State)-> State:
    messages = [HumanMessage(f"Revise this story and make it more hilarious -> {state['story']}")]
    response = LLM.invoke(messages)
    state["updated_story"] = response.content

    return state
    
def finalize_story(state: State)-> State:
    messages = [HumanMessage(f"Make final draft of this story, at the end write 2-3 sentence summary -> {state['updated_story']}")]
    response = LLM.invoke(messages)
    state["final_story"] = response.content

    return state



def main():
    # workflow simple (without a graph)
    # state = generate_story({"topic": "Chemicals"})
    # print(state)

    # print("====================================")
    # state = revise_story(state)
    # print(state["updated_story"])

    # print("====================================")
    # state = finalize_story(state)
    # print(state["final_story"])



    # workflow with graph's states
    workflow = StateGraph(State)
    workflow.add_node("first_node", generate_story)
    workflow.add_node("second_node", revise_story)
    workflow.add_node("third_node", finalize_story)

    workflow.add_edge(START, "first_node")
    workflow.add_edge("first_node", "second_node")
    workflow.add_edge("second_node", "third_node")
    workflow.add_edge("third_node", END)

    chain = workflow.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/prompting_chain.png", "wb") as f:
        f.write(graph_png)


    response = chain.invoke({"topic": "Naruto's learning path till when they fight with Sasuke after Hitachi returned for Naruto"})
    print(response)



if __name__ == "__main__":
    main()