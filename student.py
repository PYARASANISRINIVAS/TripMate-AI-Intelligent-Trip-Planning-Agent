
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv


load_dotenv()


# 1. State
class StudentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    name: str
    age: int
    course: str

    reply: str
    status: str
    confirmed: bool


# 2. Structured response from LLM
class StudentDecision(BaseModel):
    reply: str = Field(description="The next conversational response to the user")

    name: str | None = None
    age: int | None = None
    course: str | None = None

    status: str = Field(
        description="collecting, confirmation, or completed"
    )


# 3. LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

student_llm = llm.with_structured_output(StudentDecision)


# 4. Supervisor
def supervisor(state: StudentState):

    prompt = """
    You are a friendly student registration assistant.

    Collect these details:
    1. Name
    2. Age
    3. Course

    Rules:
    - Understand natural language responses.
    - Do not ask for information already provided.
    - Ask only one question at a time.
    - Update the details using the conversation.
    - When all three details are available, show a summary
      and ask the user to confirm.
    - If the user corrects something, update it and show
      the corrected summary again.
    - Mark status as completed only when the user confirms
      the summary.
    - Never assume missing details.

    Current student details:
    Name: {name}
    Age: {age}
    Course: {course}

    If the user has not started, greet them and ask for their name.
    """

    messages = [
        SystemMessage(content=prompt.format(
            name=state.get("name", "Not provided"),
            age=state.get("age", "Not provided"),
            course=state.get("course", "Not provided")
        )),
        *state.get("messages", [])
    ]

    decision = student_llm.invoke(messages)

    update = {
        "reply": decision.reply,
        "status": decision.status
    }

    if decision.name is not None:
        update["name"] = decision.name

    if decision.age is not None:
        update["age"] = decision.age

    if decision.course is not None:
        update["course"] = decision.course

    return update


# 5. Ask user using interrupt
def ask_user(state: StudentState):

    answer = interrupt(state["reply"])

    return {
        "messages": [HumanMessage(content=str(answer))]
    }


# 6. Routing
def route_after_supervisor(state: StudentState):

    if state.get("status") == "completed":
        return END

    return "ask_user"


# 7. Build graph
builder = StateGraph(StudentState)

builder.add_node("supervisor", supervisor)
builder.add_node("ask_user", ask_user)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "ask_user": "ask_user",
        END: END
    }
)

builder.add_edge("ask_user", "supervisor")

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)


# 8. Run application
config = {
    "configurable": {
        "thread_id": "student_001"
    }
}

result = graph.invoke(
    {"messages": [HumanMessage(content="Hi, I want to register.")]},
    config=config
)

while True:

    if result.get("__interrupt__"):

        question = result["__interrupt__"][0].value

        answer = input(f"\nAssistant: {question}\nYou: ")

        result = graph.invoke(
            Command(resume=answer),
            config=config
        )

    else:
        break


# 9. Display final details
print("\nRegistration completed!")

print("Name:", result.get("name"))
print("Age:", result.get("age"))
print("Course:", result.get("course"))
