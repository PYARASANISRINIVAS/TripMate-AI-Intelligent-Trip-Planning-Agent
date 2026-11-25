
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from create_pdf import generate_pdf
from tools import planning_tools, geo_location

from nodes import (
    supervisor,
    ask_user,
    route_supervisor,
    planning_node,
    route_planning,
    route_after_user
)

from state import TripState


builder = StateGraph(TripState)

builder.add_node("supervisor", supervisor)
builder.add_node("planning", planning_node)
builder.add_node("generate_pdf",generate_pdf)
builder.add_node("planning_tools", ToolNode(planning_tools))
builder.add_node("tools", ToolNode([geo_location]))
builder.add_node("ask_user", ask_user)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "planning": "planning",
        "tools": "tools",
        "ask_user": "ask_user"
    }
)

builder.add_conditional_edges("planning", route_planning, {
    "planning_tools": "planning_tools",
    "ask_user": "ask_user",
    "pdf": "pdf",
    END: END
})

builder.add_edge("planning_tools", "planning")
builder.add_edge("tools", "supervisor")

builder.add_conditional_edges("planning", route_planning, {
    "planning_tools": "planning_tools",
    "ask_user": "ask_user",
    "pdf": "pdf",
    END: END
})

checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)

png_data = graph.get_graph().draw_mermaid_png()

with open("supervisor_agent.png", "wb") as file:
    file.write(png_data)
