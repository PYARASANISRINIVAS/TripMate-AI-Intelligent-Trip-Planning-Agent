
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage
)

from langgraph.graph import END
from langgraph.types import interrupt

from state import TripState, TripDecision
from tools import geo_location, planning_tools
from prompt import SUPERVISOR_PROMPT, PLANNING_PROMPT


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

load_dotenv()

today = datetime.now(ZoneInfo("Asia/Kolkata")).date()

date_info = {
    "current_date": today.strftime("%Y-%m-%d"),
    "current_day": today.strftime("%A")
}

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


tool_call_counts = {}
counted_tool_messages = set()
MAX_TOOL_CALLS = 3
  

llm_with_tools = llm.bind_tools([geo_location])

struc_trip_llm = llm.with_structured_output(
    TripDecision,
    method="json_schema",
    strict=False
)

planning_llm = llm.bind_tools(planning_tools)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def parse_tool_output(content):
    if not isinstance(content, str):
        return content

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def select_flight(user_input, available_flights):

    if not isinstance(available_flights, list):
        return None

    user_input = user_input.strip().lower()

    if user_input.isdigit():
        index = int(user_input) - 1

        if 0 <= index < len(available_flights):
            return available_flights[index]

    for flight in available_flights:
        if isinstance(flight, dict):
            if str(flight.get("flight_number", "")).lower() == user_input:
                return flight

    return None


# --------------------------------------------------
# ASK USER
# --------------------------------------------------

def ask_user(state: TripState):

    print("Asking User!")

    answer = interrupt(state.get("reply", ""))

    return {
        "user_input": str(answer),
        "messages": [HumanMessage(content=str(answer))],
        "waiting_for_user": False
    }


# --------------------------------------------------
# SUPERVISOR
# --------------------------------------------------

def supervisor(state: TripState):

    print("Supervisor Started!")

    current_details = {
        "destination": state.get("destination", "Not provided"),
        "country": state.get("country", "Not provided"),
        "latitude": state.get("latitude", "Not provided"),
        "longitude": state.get("longitude", "Not provided"),
        "origin": state.get("origin", "Not provided"),
        "start_date": state.get("start_date", "Not provided"),
        "end_date": state.get("end_date", "Not provided"),
        "budget": state.get("budget", "Not provided"),
        "currency": state.get("currency", "Not provided"),
        "travelers": state.get("travelers", "Not provided"),
        "accommodation": state.get("accommodation", "Not provided"),
        "transportation": state.get("transportation", "Not provided"),
        "preferences": state.get("preferences", "Not provided")
    }

    location_result = state.get("location_result", "Not available")

    system_prompt = SUPERVISOR_PROMPT.format(
        current_date=date_info["current_date"],
        current_day=date_info["current_day"],
        location_result=location_result,
        **current_details
    )

    user_input = state.get("user_input", "").strip()

    history = state.get("messages", [])

    # Avoid duplicate latest user message
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]

    messages = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=user_input)
    ]

    # Geolocation tool
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print("Supervisor tool calling!")

        return {
            "messages": [response]
        }

    # Extract trip details
    structured_response = struc_trip_llm.invoke([
        SystemMessage(content="""
            Extract trip information using the TripDecision schema.

            Consider previous conversation and latest user input.
            Preserve previously collected trip details.
            Set confirmed=True only when the user explicitly
            confirms the displayed trip summary.
        """),
        *history,
        HumanMessage(content=f"Latest user input: {user_input}"),
        response
    ])

    # Handle explicit confirmation
    affirmative_responses = {
        "yes", "yeah", "yep", "ok", "okay",
        "confirm", "confirmed", "correct",
        "proceed", "go ahead"
    }

    if (
        state.get("status") == "confirmation"
        and user_input.lower() in affirmative_responses
    ):
        structured_response.confirmed = True
        structured_response.status = "completed"

    print("Supervisor structured output!")
    print("Confirmed:", structured_response.confirmed)
    print("Status:", structured_response.status)

    # Handoff to Planning Agent
    if (
        structured_response.confirmed
        and structured_response.status == "completed"
    ):

        messages_to_remove = [
            RemoveMessage(id=message.id)
            for message in state.get("messages", [])
            if message.id
        ]

        return {
            "messages": messages_to_remove,
            **structured_response.model_dump(exclude_none=True),
            "planning_step": "flights",
            "waiting_for_user": False,
            "reply": ""
        }

    return {
        "messages": [response],
        **structured_response.model_dump(exclude_none=True),
        "reply": response.content,
        "waiting_for_user": True
    }


# --------------------------------------------------
# PLANNING AGENT
# --------------------------------------------------


def planning_node(state: TripState):

    print("Planning Agent Started!")

    step = state.get("planning_step")
    user_input = state.get("user_input", "").strip()

    updates = {}

    # Save tool results and count unique executions
    for message in state.get("messages", []):

        if not isinstance(message, ToolMessage):
            continue

        message_id = message.tool_call_id or message.id

        if message_id not in counted_tool_messages:
            tool_call_counts[message.name] = (
                tool_call_counts.get(message.name, 0) + 1
            )
            counted_tool_messages.add(message_id)

        output = parse_tool_output(message.content)

        if message.name == "get_flights":
            updates["available_flights"] = output

        elif message.name == "get_hotel_list":
            updates["available_hotels"] = output

        elif message.name == "get_tourist_places":
            updates["tourist_places"] = output

        elif message.name == "get_weather":
            updates["weather_data"] = output

    # Get available data and limit results
    flights = updates.get("available_flights", state.get("available_flights", []))
    hotels = updates.get("available_hotels", state.get("available_hotels", []))
    places = updates.get("tourist_places", state.get("tourist_places", []))
    weather = updates.get("weather_data", state.get("weather_data", []))

    flights = flights[:5] if isinstance(flights, list) else flights
    hotels = hotels[:5] if isinstance(hotels, list) else hotels
    places = places[:5] if isinstance(places, list) else places

    # Flight selection
    if step == "flight_confirmation" and user_input:
        selected = select_flight(user_input, flights)

        if selected is None:
            return {
                **updates,
                "reply": "Please enter a valid flight number or option number.",
                "waiting_for_user": True
            }

        updates["selected_flight"] = selected
        updates["planning_step"] = "hotels"
        step = "hotels"

    # Hotel selection
    elif step == "hotel_confirmation" and user_input:
        selected = None

        if user_input.isdigit() and isinstance(hotels, list):
            index = int(user_input) - 1
            if 0 <= index < len(hotels):
                selected = hotels[index]

        if selected is None:
            return {
                **updates,
                "reply": "Please enter a valid hotel option number.",
                "waiting_for_user": True
            }

        updates["selected_hotel"] = selected
        updates["planning_step"] = "places"
        step = "places"

    # Tourist place selection
    elif step == "places_confirmation" and user_input:
        if user_input.lower() in {"none", "no", "nothing", "none of them"}:
            selected_places = []
        else:
            selected_places = [
                places[int(value.strip()) - 1]
                for value in user_input.split(",")
                if value.strip().isdigit()
                and 0 <= int(value.strip()) - 1 < len(places)
            ]

        updates["selected_places"] = selected_places
        updates["planning_step"] = "additional_requirements"
        step = "additional_requirements"

    # Additional requirements
    elif step == "additional_requirements" and user_input:
        updates["additional_requirements"] = (
            [] if user_input.lower() in {
                "no", "nothing", "nothing else", "none", "no thanks"
            }
            else [user_input]
        )
        updates["planning_step"] = "weather"
        step = "weather"

    # Build trip context
    trip_details = f"""
    Destination: {state.get('destination')}
    Country: {state.get('country')}
    Origin: {state.get('origin')}
    Start Date: {state.get('start_date')}
    End Date: {state.get('end_date')}
    Budget: {state.get('budget')}
    Currency: {state.get('currency')}
    Travelers: {state.get('travelers')}
    Accommodation: {state.get('accommodation')}
    Transportation: {state.get('transportation')}
    Preferences: {state.get('preferences')}

    Current Stage: {step}

    Available Flights: {flights}
    Selected Flight: {updates.get('selected_flight', state.get('selected_flight'))}

    Available Hotels: {hotels}
    Selected Hotel: {updates.get('selected_hotel', state.get('selected_hotel'))}

    Tourist Places: {places}
    Selected Places: {updates.get('selected_places', state.get('selected_places'))}

    Additional Requirements:
    {updates.get('additional_requirements', state.get('additional_requirements'))}

    Weather: {weather}
    """

    recent_messages = [
        msg for msg in state.get("messages", [])
        if not isinstance(msg, ToolMessage)
        and not getattr(msg, "tool_calls", None)
    ][-3:]

    messages = [
        SystemMessage(content=PLANNING_PROMPT),
        SystemMessage(content=trip_details),
        *recent_messages
    ]

    # Restrict tools based on their call counts
    available_tools = [
        tool for tool in planning_tools
        if tool_call_counts.get(tool.name, 0) < MAX_TOOL_CALLS
    ]

    llm_with_tools = planning_llm.bind_tools(available_tools)

    print("Tool call counts:", tool_call_counts)

    response = llm_with_tools.invoke(messages)

    updates["messages"] = [response]

    if response.tool_calls:
        updates["reply"] = ""
        updates["waiting_for_user"] = False
    else:
        updates["reply"] = response.content
        updates["waiting_for_user"] = bool(response.content)

        if step == "flights":
            updates["planning_step"] = "flight_confirmation"
        elif step == "hotels":
            updates["planning_step"] = "hotel_confirmation"
        elif step == "places":
            updates["planning_step"] = "places_confirmation"

    return updates


# --------------------------------------------------
# SUPERVISOR ROUTING
# --------------------------------------------------

def route_supervisor(state: TripState):

    print("Supervisor Routing!")

    if (
        state.get("confirmed") is True
        and state.get("status") == "completed"
    ):
        return "planning"

    messages = state.get("messages", [])

    if messages and getattr(messages[-1], "tool_calls", None):
        return "tools"

    return "ask_user"


# --------------------------------------------------
# PLANNING ROUTING
# --------------------------------------------------

def route_planning(state: TripState):

    print("Planning Routing!")

    messages = state.get("messages", [])

    if messages and getattr(messages[-1], "tool_calls", None):
        return "planning_tools"

    if state.get("planning_step") == "completed":
        return "pdf"

    if state.get("waiting_for_user", False):
        return "ask_user"

    return END


# --------------------------------------------------
# ROUTING AFTER USER RESPONSE
# --------------------------------------------------

def route_after_user(state: TripState):

    step = state.get("planning_step")

    planning_steps = {
        "flights",
        "flight_confirmation",
        "hotels",
        "hotel_confirmation",
        "places",
        "places_confirmation",
        "additional_requirements",
        "weather",
        "finalize"
    }

    if step in planning_steps:
        return "planning"

    return "supervisor"
