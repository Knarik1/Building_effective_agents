from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel
from typing import List, TypedDict, Annotated
import operator
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class Section(BaseModel):
    name: str
    description: str 

class Sections(BaseModel):
    sections: list[Section] 


LLM = ChatOllama(model="llama3.2")
PLANNER = LLM.with_structured_output(Sections) 


class State(TypedDict):
    topic: str 
    sections: List[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section 
    completed_sections: Annotated[list, operator.add]


# nodes
def orchestrator(state: State):
    sections_list = PLANNER.invoke([
        SystemMessage("You are a planner. You generate subtasks to complete a task."),
        HumanMessage(f"Generate subtasks based on topic {state['topic']}")
    ])

    return {"sections": sections_list.sections}


def llm_call(state: WorkerState):
    completed_section = LLM.invoke([
        SystemMessage(content="You are a tasl completer."),
        HumanMessage(content=f"Generate a task completion based on topic name {state['section'].name} and description {state['section'].description}")
    ])

    return {"completed_sections": [completed_section.content]}


def syntesizer(state: State):  
    all_sections_together = "\n\n -- \n\n".join(state["completed_sections"])

    return {"final_report": all_sections_together}  


def assign_workers(state: State):
    return [Send("llm_call", {"section": section}) for section in state["sections"]]     


def main():
    workflow = StateGraph(State)
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("llm_call", llm_call)
    workflow.add_node("syntesizer", syntesizer)

    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges("orchestrator", assign_workers)
    workflow.add_edge("llm_call", "syntesizer")
    workflow.add_edge("syntesizer", END)

    chain = workflow.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/orchestrator.png", "wb") as f:
        f.write(graph_png)


    output = chain.invoke({"topic": "Create a report on LLM scaling laws"})
    print(output["final_report"])
    



if __name__ == "__main__":
    main()