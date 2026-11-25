
import streamlit as st
import uuid
import os

from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import graph


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    .trip-label {
        font-size: 12px;
        color: #888888;
        margin-bottom: 3px;
    }

    .trip-value {
        font-size: 14px;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    div[data-testid="stChatMessage"] {
        padding: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = False

if "interrupt_prompt" not in st.session_state:
    st.session_state.interrupt_prompt = None


# ==================================================
# LANGGRAPH CONFIGURATION
# ==================================================

def get_config():

    return {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


def get_graph_state():

    return graph.get_state(get_config()).values


# ==================================================
# ACTIVITY MESSAGES
# ==================================================

NODE_ACTIVITIES = {
    "supervisor": "Understanding your travel requirements...",
    "planning": "Planning your trip...",
    "tools": "Collecting location details...",
    "planning_tools": "Processing travel information...",
    "ask_user": "Waiting for your response...",
    "pdf_node": "Generating your itinerary PDF..."
}

TOOL_ACTIVITIES = {
    "geo_location": "Collecting location details...",
    "get_flights": "Fetching available flights...",
    "get_hotel_list": "Searching available hotels...",
    "get_tourist_places": "Finding tourist attractions...",
    "get_weather": "Checking weather conditions..."
}


# ==================================================
# EXTRACT MESSAGE CONTENT
# ==================================================

def extract_message_content(message):

    # Ignore tool messages
    if isinstance(message, ToolMessage):
        return ""

    # Ignore AI messages requesting tool execution
    if isinstance(message, AIMessage) and message.tool_calls:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    # Handle multimodal/list content safely
    if isinstance(content, list):

        parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))

        return "\n".join(parts).strip()

    return ""


# ==================================================
# PROCESS USER INPUT
# ==================================================

def process_user_input(user_input, status):

    config = get_config()

    previous_state = get_graph_state()
    previous_reply = previous_state.get("reply", "")

    # IMPORTANT:
    # Resume an interrupted graph with Command.
    # Otherwise, send the user's message as HumanMessage.

    if st.session_state.awaiting_input:

        graph_input = Command(resume=user_input)

    else:

        graph_input = {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_input": user_input
        }

    assistant_reply = ""
    interrupts = []

    # Stream graph updates
    for chunk in graph.stream(
        graph_input,
        config=config,
        stream_mode="updates"
    ):

        # ------------------------------------------
        # INTERRUPT HANDLING
        # ------------------------------------------

        if "__interrupt__" in chunk:

            interrupt_data = chunk["__interrupt__"]

            if isinstance(interrupt_data, (list, tuple)):
                interrupts.extend(interrupt_data)
            else:
                interrupts.append(interrupt_data)

            continue

        # ------------------------------------------
        # PROCESS NODE UPDATES
        # ------------------------------------------

        for node_name, node_update in chunk.items():

            if not isinstance(node_update, dict):
                continue

            # Show node activity
            activity = NODE_ACTIVITIES.get(node_name)

            if activity:

                status.update(
                    label=activity,
                    state="running"
                )

                status.write(activity)

            # Capture reply directly from node
            node_reply = node_update.get("reply")

            if node_reply:
                assistant_reply = str(node_reply)

            # Process updated messages
            node_messages = node_update.get("messages", [])

            if not isinstance(node_messages, list):
                node_messages = [node_messages]

            for message in node_messages:

                # Detect LLM tool calls
                if isinstance(message, AIMessage):

                    for tool_call in message.tool_calls:

                        tool_name = tool_call.get("name")

                        activity = TOOL_ACTIVITIES.get(
                            tool_name,
                            "Processing your request..."
                        )

                        status.update(
                            label=activity,
                            state="running"
                        )

                        status.write(activity)

                    # Capture normal AI response
                    message_content = extract_message_content(message)

                    if message_content:
                        assistant_reply = message_content

                # Detect completed tool calls
                elif isinstance(message, ToolMessage):

                    tool_name = message.name

                    if tool_name in TOOL_ACTIVITIES:

                        status.write(
                            f"Completed: {TOOL_ACTIVITIES[tool_name]}"
                        )

            # --------------------------------------
            # PLANNING STAGE
            # --------------------------------------

            planning_step = node_update.get("planning_step")

            if planning_step == "flight_confirmation":
                status.write("Flight options are ready.")

            elif planning_step == "hotel_confirmation":
                status.write("Hotel options are ready.")

            elif planning_step == "places_confirmation":
                status.write("Tourist attractions are ready.")

            elif planning_step == "additional_requirements":
                status.write("Collecting additional requirements...")

            elif planning_step == "weather":
                status.write("Preparing weather information...")

            elif planning_step == "completed":
                status.write("Travel itinerary completed.")

    # ==================================================
    # GET LATEST GRAPH STATE
    # ==================================================

    state = get_graph_state()

    # ==================================================
    # HANDLE INTERRUPTS
    # ==================================================

    if interrupts:

        st.session_state.awaiting_input = True

        interrupt_value = interrupts[0].value

        st.session_state.interrupt_prompt = interrupt_value

        if isinstance(interrupt_value, str):
            assistant_reply = interrupt_value
        else:
            assistant_reply = str(interrupt_value)

        status.update(
            label="Waiting for your response",
            state="complete"
        )

    else:

        st.session_state.awaiting_input = False
        st.session_state.interrupt_prompt = None

        # Use reply only if it has changed
        current_reply = state.get("reply", "")

        if not assistant_reply and current_reply != previous_reply:
            assistant_reply = current_reply

        # Fallback: inspect the latest AI message
        if not assistant_reply:

            messages = state.get("messages", [])

            for message in reversed(messages):

                message_content = extract_message_content(message)

                if message_content:
                    assistant_reply = message_content
                    break

        status.update(
            label="Completed",
            state="complete"
        )

    # ==================================================
    # SAVE ASSISTANT RESPONSE
    # ==================================================

    if assistant_reply:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_reply
        })

    else:

        print("\n========== DEBUG ==========")
        print("No assistant response found.")
        print("User input:", user_input)
        print("Planning step:", state.get("planning_step"))
        print("Reply:", state.get("reply"))
        print("Messages:", state.get("messages"))
        print("Next graph nodes:", graph.get_state(config).next)
        print("===========================\n")


# ==================================================
# START NEW TRIP
# ==================================================

def start_new_trip():

    st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.chat_history = []

    st.session_state.awaiting_input = False

    st.session_state.interrupt_prompt = None


# ==================================================
# TRIP DETAILS
# ==================================================

def display_trip_details(state):

    trip_fields = [
        ("Origin", "origin"),
        ("Destination", "destination"),
        ("Country", "country"),
        ("Start Date", "start_date"),
        ("End Date", "end_date"),
        ("Budget", "budget"),
        ("Currency", "currency"),
        ("Travelers", "travelers"),
        ("Accommodation", "accommodation"),
        ("Transportation", "transportation"),
        ("Preferences", "preferences"),
        ("Additional Requirements", "additional_requirements")
    ]

    existing_fields = [
        (label, state.get(key))
        for label, key in trip_fields
        if state.get(key) not in (None, "", [])
    ]

    if not existing_fields:

        st.info("Start a conversation to enter your trip details.")
        return

    columns = st.columns(3)

    for index, (label, value) in enumerate(existing_fields):

        with columns[index % 3]:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div class="trip-label">{label}</div>
                    <div class="trip-value">{value}</div>
                    """,
                    unsafe_allow_html=True
                )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("✈️ Travel Assistant")

    st.divider()

    if st.button("＋ New Trip", use_container_width=True):

        start_new_trip()
        st.rerun()

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "💬 Chat",
            "📋 Trip Details",
            "🗺️ Itinerary"
        ],
        label_visibility="collapsed"
    )


# ==================================================
# CHAT PAGE
# ==================================================

if page == "💬 Chat":

    st.title("💬 Chat")

    st.caption("Plan your trip through a conversation.")

    # Display conversation history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Enter your message...")

    if user_input:

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.status(
                "Preparing your request...",
                expanded=True
            ) as status:

                try:

                    process_user_input(
                        user_input,
                        status
                    )

                except Exception as e:

                    status.update(
                        label="Something went wrong",
                        state="error"
                    )

                    st.error(str(e))

                    print("STREAMLIT GRAPH ERROR:", repr(e))

        st.rerun()


# ==================================================
# TRIP DETAILS PAGE
# ==================================================

elif page == "📋 Trip Details":

    st.title("📋 Trip Details")

    try:

        state = get_graph_state()

        if not state:

            st.info("Start a conversation to enter your trip details.")

        else:

            display_trip_details(state)

            st.divider()

            st.subheader("Selected Options")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**✈️ Selected Flight**")

                flight = state.get("selected_flight")

                if flight:
                    st.json(flight, expanded=False)
                else:
                    st.caption("Not selected yet.")

            with col2:

                st.markdown("**🏨 Selected Hotel**")

                hotel = state.get("selected_hotel")

                if hotel:
                    st.json(hotel, expanded=False)
                else:
                    st.caption("Not selected yet.")

            st.markdown("**📍 Selected Tourist Places**")

            places = state.get("selected_places", [])

            if places:

                for place in places:
                    st.write(f"• {place}")

            else:

                st.caption("No places selected yet.")

    except Exception as e:

        st.error(f"Unable to retrieve trip details: {str(e)}")


# ==================================================
# ITINERARY PAGE
# ==================================================

elif page == "🗺️ Itinerary":

    st.title("🗺️ Your Itinerary")

    try:

        state = get_graph_state()

        if not state:

            st.info("Your itinerary will appear here after trip planning.")

        else:

            if state.get("planning_step") == "completed":

                final_plan = (
                    state.get("final_plan")
                    or state.get("reply")
                )

                if final_plan:
                    st.markdown(final_plan)

            else:

                st.info(
                    "Your itinerary is not ready yet. "
                    "Continue the conversation."
                )

            # PDF download
            pdf_path = state.get("pdf_path")

            if pdf_path and os.path.exists(pdf_path):

                st.divider()

                with open(pdf_path, "rb") as pdf_file:

                    st.download_button(
                        label="⬇️ Download Itinerary PDF",
                        data=pdf_file.read(),
                        file_name="travel_itinerary.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            elif state.get("planning_step") == "completed":

                st.warning(
                    "The itinerary is ready, but the PDF "
                    "has not been generated yet."
                )

    except Exception as e:

        st.error(f"Unable to load itinerary: {str(e)}")
