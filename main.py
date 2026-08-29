from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

MODEL_NAME = "qwen2.5:1.5b-instruct"
SYS_PROMPT = "You are an helpful assistant. If I asked you to multiply 2 numbers dont do it by yourself, just call the tool that you have"


class Response(BaseModel):
    height: int 
    width: int


def multiply_tool(a: int, b: int):
    return a * b    


def main():
    # define the model
    llm = ChatOllama(model=MODEL_NAME)

    # define the prompt
    prompt = "What is the height and width of Effel tower by meter, just imagine, use your tool to multiple the parameters?"
    messages = [HumanMessage(prompt), SystemMessage(SYS_PROMPT)]

    # model calling
    # response = llm.invoke(prompt)
    # print(response)

    # llm_structured = llm.with_structured_output(Response)
    # response = llm_structured.invoke(messages)
    # print(response)

    llm_with_tools = llm.bind_tools([multiply_tool])
    response = llm_with_tools.invoke(prompt)

    # this isnt working properly I guess
    # llm_with_tools = llm.bind_tools([multiply_tool]).with_structured_output(Response)
    # response = llm_with_tools.invoke(prompt)
    # print(multiply_tool(response.height, response.width))


    if response.tool_calls:
        print("============== Tool calling ====================")
        print(response.tool_calls)
        print("\n\n")

    print(response.content)    

if __name__ == "__main__":
    main()