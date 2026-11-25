
from pydantic import BaseModel, Field
from typing import List, Any, TypedDict, Annotated, Literal,Optional
from langgraph.graph.message import add_messages


# Planning stages
PlanningStep = Literal[
    "weather",
    "flights",
    "flight_confirmation",
    "hotels",
    "hotel_confirmation",
    "places",
    "places_confirmation",
    "finalize",
    "completed"
]


class TripState(TypedDict):

    # Conversation history
    messages: Annotated[List, add_messages]

    # Geolocation
    location_result: dict[str, Any]

    # Trip details
    destination: str
    country: str
    latitude: float
    longitude: float

    origin: str
    start_date: str
    end_date: str

    budget: float
    currency: str
    travelers: int

    accommodation: str
    transportation: str
    preferences: list[str]

    # Conversation control
    reply: str
    user_input: str
    status: str
    confirmed: bool

    # Planning progress
    planning_step: PlanningStep
    waiting_for_user: bool

    # Retrieved data
    weather_data: dict[str, Any]
    available_flights: list
    available_hotels: list
    tourist_places: list

    # User selections
    selected_flight: dict[str, Any]
    selected_hotel: dict[str, Any]
    selected_places: list

    # Final itinerary
    final_plan: str
    pdf_path: str


class TripDecision(BaseModel):


    reply: str = Field(
        description="The next conversational response to the user"
    )

    destination: str | None = None
    origin: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    start_date: str | None = None
    end_date: str | None = None

    budget: float | None = None
    currency: str | None = None
    travelers: int | None = None

    accommodation: str | None = None
    transportation: str | None = None
    preferences: list[str] | None = None

    status: Literal[
        "collecting",
        "confirmation",
        "completed"
    ]

    confirmed: bool = False





class PlanningDecision(BaseModel):

    reply: str = Field(
        description="Response to display to the user"
    )

    planning_step: Literal[
        "weather",
        "flights",
        "flight_confirmation",
        "hotels",
        "hotel_confirmation",
        "places",
        "places_confirmation",
        "finalize",
        "completed"
    ]

    action: Literal[
        "call_tool",
        "ask_user",
        "continue",
        "finish"
    ]

    selected_flight: Optional[dict] = None
    selected_hotel: Optional[dict] = None
    selected_places: Optional[list] = None