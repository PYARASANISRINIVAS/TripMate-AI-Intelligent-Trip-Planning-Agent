
import streamlit as st
import uuid
import os

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from agent import graph


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = False


# --------------------------------------------------
# LANGGRAPH CONFIGURATION
# --------------------------------------------------

def get_config():

    return {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


def get_graph_state():

    return graph.get_state(get_config()).values


# --------------------------------------------------
# PROCESS USER INPUT
# --------------------------------------------------

def process_user_input(user_input):

    config = get_config()

    # Resume interrupted graph
    if st.session_state.awaiting_input:

        result = graph.invoke(
            Command(resume=user_input),
            config=config
        )

    # Start or continue conversation
    else:

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ],
                "user_input": user_input
            },
            config=config
        )

    # Handle interrupt
    if result.get("__interrupt__"):

        question = result["__interrupt__"][0].value

        st.session_state.awaiting_input = True

        return question

    # Graph execution completed without interrupt
    st.session_state.awaiting_input = False

    # Get assistant response
    reply = result.get("reply", "")

    # Fallback: get latest AI response
    if not reply:

        for message in reversed(result.get("messages", [])):

            if isinstance(message, AIMessage) and not message.tool_calls:

                if isinstance(message.content, str):
                    reply = message.content
                    break

    return reply


# --------------------------------------------------
# NEW TRIP
# --------------------------------------------------

def start_new_trip():

    st.session_state.thread_id = str(uuid.uuid4())

    st.session_state.chat_history = []

    st.session_state.awaiting_input = False


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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

    # Display chat history
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Enter your message...")

    if user_input:

        # Display and save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):

            try:

                with st.spinner("Processing your request..."):

                    assistant_reply = process_user_input(user_input)

                if assistant_reply:

                    st.markdown(assistant_reply)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": str(assistant_reply)
                    })

                else:

                    st.warning("No response received from the assistant.")

            except Exception as e:

                st.error(f"Error: {str(e)}")

                print("GRAPH ERROR:", repr(e))

        st.rerun()


# ==================================================
# TRIP DETAILS PAGE
# ==================================================

elif page == "📋 Trip Details":

    st.title("📋 Trip Details")

    state = get_graph_state()

    if not state:

        st.info("Start a conversation to enter your trip details.")

    else:

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

        if existing_fields:

            columns = st.columns(3)

            for index, (label, value) in enumerate(existing_fields):

                with columns[index % 3]:

                    with st.container(border=True):

                        st.caption(label)
                        st.write(value)

        else:

            st.info("Trip details will appear here as you provide them.")

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


# ==================================================
# ITINERARY PAGE
# ==================================================

elif page == "🗺️ Itinerary":

    st.title("🗺️ Your Itinerary")

    state = get_graph_state()

    if not state:

        st.info("Your itinerary will appear here after trip planning.")

    elif state.get("planning_step") == "completed":

        final_plan = (
            state.get("final_plan")
            or state.get("reply")
        )

        if final_plan:
            st.markdown(final_plan)

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

        else:

            st.warning("PDF has not been generated yet.")

    else:

        st.info("Your itinerary is not ready yet. Continue the conversation.")
