
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from tools import planning_tools
from tools import geo_location
from nodes import supervisor,ask_user,route_supervisor,planning_node,route_planning,route_after_user,pdf_node
from prompt import SUPERVISOR_PROMPT


from state import TripDecision,TripState

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(SUPERVISOR_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


builder = StateGraph(TripState)

# Nodes
builder.add_node("supervisor", supervisor)
builder.add_node("planning", planning_node)
builder.add_node("planning_tools", ToolNode(planning_tools))
builder.add_node("tools", ToolNode([geo_location]))
builder.add_node("ask_user", ask_user)
builder.add_node("pdf_node", pdf_node)
# Start
builder.add_edge(START, "supervisor")

# Supervisor routing
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "planning": "planning",
        "tools": "tools",
        "ask_user": "ask_user"
    }
)

# Planning routing
builder.add_conditional_edges(
    "planning",
    route_planning,
    {
        "planning": "planning",
        "planning_tools": "planning_tools",
        "ask_user": "ask_user",
        "pdf_node": "pdf_node"
    }
)

# Tool loops
builder.add_edge("planning_tools", "planning")
builder.add_edge("tools", "supervisor")
builder.add_edge("pdf_node",END)
# After user responds
builder.add_conditional_edges(
    "ask_user",
    route_after_user,
    {
        "planning": "planning",
        "supervisor": "supervisor"
    }
)

# Compile
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


png_data = graph.get_graph().draw_mermaid_png()

with open("supervisor_agent.png", "wb") as file:
    file.write(png_data)