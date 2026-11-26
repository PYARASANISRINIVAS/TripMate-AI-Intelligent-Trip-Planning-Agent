
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from create_pdf import generate_pdf
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

    available_tools = (
    [geo_location]
    if tool_call_counts.get("geo_location", 0) < MAX_TOOL_CALLS
    else []
)

    response = llm.bind_tools(available_tools).invoke(messages)

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
def select_hotel(user_input, hotels):

    response = llm.invoke([
        SystemMessage(content="""
        You are a hotel selection assistant.

        Select the hotel that best matches the user's request.

        The user may:
        - Mention a hotel name.
        - Ask for the cheapest hotel.
        - Ask for the highest-rated hotel.
        - Ask for a hotel near a particular location.
        - Describe other preferences.

        Return ONLY the option number from the available hotels.
        If no suitable hotel can be identified, return NONE.
        """),
        HumanMessage(content=f"""
        Available hotels:
        {hotels}

        User request:
        {user_input}
        """)
    ])

    result = response.content.strip()

    if result.isdigit():
        index = int(result) - 1

        if 0 <= index < len(hotels):
            return hotels[index]

    return None


def planning_node(state: TripState):

    print("Planning Agent Started!")

    step = state.get("planning_step")
    user_input = state.get("user_input", "").strip()

    updates = {}

    # --------------------------------------------------
    # 1. Process tool results
    # --------------------------------------------------

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

        if (
            getattr(message, "status", "success") == "error"
            or output is None
            or output == ""
            or output == []
            or output == {}
        ):
            output = []

        if message.name == "get_flights":
            updates["available_flights"] = output

        elif message.name == "get_hotel_list":
            updates["available_hotels"] = output

        elif message.name == "get_tourist_places":
            updates["tourist_places"] = output

        elif message.name == "get_weather":
            updates["weather_data"] = output

    # --------------------------------------------------
    # 2. Retrieve available data
    # --------------------------------------------------

    flights = updates.get(
        "available_flights",
        state.get("available_flights", [])
    )

    hotels = updates.get(
        "available_hotels",
        state.get("available_hotels", [])
    )

    places = updates.get(
        "tourist_places",
        state.get("tourist_places", [])
    )

    weather = updates.get(
        "weather_data",
        state.get("weather_data", [])
    )

    flights = flights[:5] if isinstance(flights, list) else []
    hotels = hotels[:5] if isinstance(hotels, list) else []
    places = places[:5] if isinstance(places, list) else []

    # --------------------------------------------------
    # 3. Handle user responses using the LLM
    # --------------------------------------------------

    confirmation_stages = {
        "flight_confirmation": {
            "data": flights,
            "field": "selected_flight",
            "next": "hotels"
        },
        "hotel_confirmation": {
            "data": hotels,
            "field": "selected_hotel",
            "next": "places"
        },
        "places_confirmation": {
            "data": places,
            "field": "selected_places",
            "next": "additional_requirements"
        }
    }

    if step in confirmation_stages and user_input:

        config = confirmation_stages[step]

        decision_prompt = f"""
        You are an intelligent travel planning assistant.

        Current planning stage:
        {step}

        Available options:
        {config["data"]}

        Previous selections:
        Flights: {state.get("selected_flight")}
        Hotels: {state.get("selected_hotel")}
        Tourist places: {state.get("selected_places")}

        User response:
        {user_input}

        Understand the user's intent from the complete context.

        You must return ONLY valid JSON with these fields:

        {{
            "action": "select | skip | search_again | clarify | continue",
            "selected_indexes": [],
            "response": ""
        }}

        Interpret natural language without relying on fixed keywords
        or predefined phrases.

        Select the appropriate available options based on the user's
        preferences, comparisons, names, descriptions and requirements.

        Use zero-based indexes for selected options.

        Never invent options or indexes.

        If the user wants to postpone or move forward, identify that
        intent and return the appropriate action.

        If the user wants different options, return search_again.

        If the user's intent is unclear, return clarify.

        If the user wants more information about the current options,
        return continue.

        For tourist places, multiple selections are allowed.
        For flights and hotels, select only one option.

        The response must be natural and relevant to the conversation.
        """

        decision_response = llm.invoke([
            SystemMessage(content=decision_prompt)
        ])

        try:
            decision = json.loads(decision_response.content)

        except (json.JSONDecodeError, TypeError):
            return {
                **updates,
                "reply": decision_response.content,
                "waiting_for_user": True
            }

        action = decision.get("action")
        indexes = decision.get("selected_indexes", [])

        # User selected an option
        if action == "select":

            valid_indexes = [
                i for i in indexes
                if isinstance(i, int)
                and 0 <= i < len(config["data"])
            ]

            if not valid_indexes:

                return {
                    **updates,
                    "reply": decision.get("response", ""),
                    "waiting_for_user": True
                }

            selected = [
                config["data"][i] for i in valid_indexes
            ]

            if step == "places_confirmation":
                updates[config["field"]] = selected
            else:
                updates[config["field"]] = selected[0]

            return {
                **updates,
                "reply": decision.get("response", ""),
                "planning_step": config["next"],
                "waiting_for_user": False
            }

        # User wants to skip the current selection
        elif action == "skip":

            return {
                **updates,
                "reply": decision.get("response", ""),
                "planning_step": config["next"],
                "waiting_for_user": False
            }

        # User wants another search
        elif action == "search_again":

            step = {
                "flight_confirmation": "flights",
                "hotel_confirmation": "hotels",
                "places_confirmation": "places"
            }[step]

        # User needs clarification or more information
        elif action in ["clarify", "continue"]:

            return {
                **updates,
                "reply": decision.get("response", ""),
                "waiting_for_user": True
            }

    # --------------------------------------------------
    # 4. Check retry limits
    # --------------------------------------------------

    stage_config = {
        "flights": (
            "get_flights",
            "hotels"
        ),
        "hotels": (
            "get_hotel_list",
            "places"
        ),
        "places": (
            "get_tourist_places",
            "weather"
        ),
        "weather": (
            "get_weather",
            "completed"
        )
    }

    stage_data = {
        "flights": flights,
        "hotels": hotels,
        "places": places,
        "weather": weather
    }

    if step in stage_config:

        tool_name, next_stage = stage_config[step]

        if (
            tool_call_counts.get(tool_name, 0) >= MAX_TOOL_CALLS
            and not stage_data[step]
        ):

            return {
                **updates,
                "reply": (
                    f"Unable to retrieve {step} after "
                    f"{MAX_TOOL_CALLS} attempts. "
                    "Let's continue with the next stage."
                ),
                "planning_step": next_stage,
                "waiting_for_user": False
            }

    # --------------------------------------------------
    # 5. Build trip context
    # --------------------------------------------------

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
    Selected Flight: {state.get('selected_flight')}

    Available Hotels: {hotels}
    Selected Hotel: {state.get('selected_hotel')}

    Tourist Places: {places}
    Selected Places: {state.get('selected_places')}

    Additional Requirements:
    {state.get('additional_requirements')}

    Weather: {weather}

    Latest User Input:
    {user_input}
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

    # --------------------------------------------------
    # 6. Bind available tools
    # --------------------------------------------------

    available_tools = [
        tool for tool in planning_tools
        if tool_call_counts.get(tool.name, 0) < MAX_TOOL_CALLS
    ]

    print("Tool call counts:", tool_call_counts)
    print("Available tools:", [tool.name for tool in available_tools])

    # Use the unbound base LLM here
    llm_with_tools = llm.bind_tools(available_tools)

    response = llm_with_tools.invoke(messages)

    updates["messages"] = [response]

    # --------------------------------------------------
    # 7. Handle tool calls and responses
    # --------------------------------------------------

    if response.tool_calls:

        updates["reply"] = ""
        updates["waiting_for_user"] = False

    else:

        updates["reply"] = response.content

        next_stages = {
            "flights": "flight_confirmation",
            "hotels": "hotel_confirmation",
            "places": "places_confirmation",
            "weather": "completed"
        }

        if step in next_stages:

            updates["planning_step"] = next_stages[step]

            updates["waiting_for_user"] = (
                step != "weather"
            )

    print("Planning Step:", updates.get("planning_step", step))
    print("Waiting for User:", updates.get("waiting_for_user"))

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

    messages = state.get("messages", [])

    if messages and getattr(messages[-1], "tool_calls", None):
        return "planning_tools"

    if state.get("planning_step") == "completed":
        return "pdf_node"

    if state.get("waiting_for_user", False):
        return "ask_user"

    return "planning"



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


def pdf_node(state: TripState):
    markdown_content = state.get("reply", "")

    pdf_path = generate_pdf(markdown_content)

    return {
        "pdf_path": pdf_path
    }