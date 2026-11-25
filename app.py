
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import graph

config = {
    "configurable": {
        "thread_id": "trip_002"
    }
}


print("Welcome to TripPilot AI!")

query = input("\nYou: ")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(content=query)
        ],
        "user_input": query
    },
    config=config
)


# Handle multiple interrupts
while True:

    if result.get("__interrupt__"):

        question = result["__interrupt__"][0].value

        answer = input(
            f"\nTripPilot AI: {question}\nYou: "
        )

        result = graph.invoke(
            Command(resume=answer),
            config=config
        )

    else:
        break