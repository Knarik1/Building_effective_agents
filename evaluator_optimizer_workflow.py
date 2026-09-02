from langchain_ollama.chat_models import ChatOllama
from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage


# structured output
class Feedback(BaseModel):
    grade: Literal["funny", "not_funny"] = Field(description="You should grade the topic based on this 2 evaluations")
    feedback: str = Field(description="If the joke is not_funny, give a feedback how to improve it")

LLM = ChatOllama(model="llama3.2")
EVALUATOR = LLM.with_structured_output(Feedback)    


class State(TypedDict):
    topic: str 
    joke: str 
    feedback: str 
    decision: Literal["funny", "not_funny"]


# nodes
def llm_call_generator(state: State):
    if state.get("feedback"):
        joke = LLM.invoke([
                SystemMessage("You are a creative joke creator"),
                HumanMessage(f"Generate a joke based on topic {state['topic']} but take into accout feedback {state['feedback']}")
            ])
    else:
        joke = LLM.invoke([
            SystemMessage("You are a creative joke creator"),
            HumanMessage(f"Generate a joke based on topic {state['topic']}")
        ])

    return {"joke": joke.content}


def llm_call_evaluator(state: State):
    response = EVALUATOR.invoke([
        SystemMessage("You are helpful joke critic on creativity"),
        HumanMessage(f"Generate useful critic about the joke {state['joke']}.")
    ])

    return {"decision": response.grade, "feedback": response.feedback}   


def feedback_loop(state: State):
    if state["decision"] == "funny":
        print("==================== Finish")
        return "Finish"
    else: 
        print("==================== Loop")
        return "Loop" 


def main():
    

    workflow = StateGraph(State)
    workflow.add_node("generator", llm_call_generator)
    workflow.add_node("evaluator", llm_call_evaluator)

    workflow.add_edge(START, "generator")
    workflow.add_edge("generator", "evaluator")
    workflow.add_conditional_edges("evaluator", feedback_loop, {"Finish": END, "Loop": "generator"})
    workflow.add_edge("evaluator", END)

    chain = workflow.compile()

    # visualize
    graph_png = chain.get_graph().draw_mermaid_png()
    with open("./images/evaluator_optimizer.png", "wb") as f:
        f.write(graph_png)

    response = chain.invoke({"topic": "Naruto"})
    print(response)

if __name__ == "__main__":
    main()