
import streamlit as st
import uuid

from langgraph.types import Command
from agent import graph


# 1. Page configuration
st.set_page_config(
    page_title="TripPilot AI",
    page_icon="✈️",
    layout="wide"
)


# 2. Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "waiting" not in st.session_state:
    st.session_state.waiting = False


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# 3. Sidebar
with st.sidebar:

    st.title("✈️ TripPilot AI")
    st.caption("Your AI Travel Assistant")

    page = st.radio(
        "Navigation",
        [
            "💬 Chat",
            "🧳 Trip Details",
            "🗺️ Itinerary"
        ]
    )

    st.divider()

    if st.button("➕ New Trip", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.waiting = False

        st.rerun()


# 4. Chat page
if page == "💬 Chat":

    st.title("✈️ TripPilot AI")
    st.caption("Your personal AI travel planner")

    # Display chat history
    for role, message in st.session_state.chat_history:

        with st.chat_message(role):
            st.markdown(message)

    # User input
    user_input = st.chat_input("Where would you like to travel?")

    if user_input:

        st.session_state.chat_history.append(
            ("user", user_input)
        )

        try:

            with st.spinner("Planning your trip..."):

                # Resume interrupted graph
                if st.session_state.waiting:

                    result = graph.invoke(
                        Command(resume=user_input),
                        config=config
                    )

                # Start graph
                else:

                    result = graph.invoke(
                        {
                            "messages": [],
                            "user_input": user_input
                        },
                        config=config
                    )

                # Check for interrupt
                interrupts = result.get("__interrupt__", [])

                if interrupts:

                    response = interrupts[0].value
                    st.session_state.waiting = True

                else:

                    st.session_state.waiting = False

                    response = (
                        result.get("reply")
                        or result.get("final_plan")
                    )

                    # Get latest assistant message if needed
                    if not response:

                        messages = result.get("messages", [])

                        response = next(
                            (
                                msg.content
                                for msg in reversed(messages)
                                if getattr(msg, "content", None)
                                and not getattr(msg, "tool_calls", [])
                            ),
                            "I'm processing your request."
                        )

            # Save assistant response
            st.session_state.chat_history.append(
                ("assistant", str(response))
            )

        except Exception as e:

            st.error(f"Something went wrong: {e}")

        st.rerun()


# 5. Trip details page
elif page == "🧳 Trip Details":

    st.title("🧳 Trip Details")

    state = graph.get_state(config).values

    fields = [
        "destination",
        "country",
        "origin",
        "start_date",
        "end_date",
        "budget",
        "currency",
        "travelers",
        "accommodation",
        "transportation"
    ]

    for field in fields:

        value = state.get(field, "Not provided")

        st.write(
            f"**{field.replace('_', ' ').title()}:** {value}"
        )


# 6. Itinerary page
elif page == "🗺️ Itinerary":

    st.title("🗺️ Your Travel Itinerary")

    state = graph.get_state(config).values

    final_plan = state.get("final_plan")

    if final_plan:
        st.markdown(final_plan)

    else:
        st.info("Your itinerary will appear here after planning.")
