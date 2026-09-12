import json
import streamlit as st
from recommender import ai_chatbot

st.set_page_config(page_title="Hostel Recommender", page_icon="🏠")
st.title("Hostel Recommendation Chatbot")
st.caption("Tell me what you're looking for in a hostel — room type, budget, distance to campus. Or use Quick Search below.")

# Initialize chat history (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []


def ask(prompt: str):
    """Send a prompt to the chatbot and store both sides of the exchange."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Searching hostels..."):
        reply = ai_chatbot(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def copy_button(text: str, key: str):
    """Minimal icon-only copy button, styled like Claude's copy affordance.

    Renders directly into the page via st.html (unsafe_allow_javascript=True)
    instead of components.html, so it doesn't spin up a separate iframe per
    button. That matters here because this is called once per user message,
    on every rerun of the whole message history — with components.html each
    of those was a full nested browser context; with st.html it's just DOM.

    Uses a Unicode glyph instead of an inline <svg> for the icon, since
    st.html runs content through DOMPurify and inline SVG markup gets
    stripped, leaving an empty (but still clickable) button.
    """
    # Put the text inside a <script> block (not an HTML attribute) so quotes,
    # apostrophes, and newlines in the message can't break the markup.
    js_text = json.dumps(text).replace("</", "<\\/")
    st.html(f"""
        <button id="{key}" title="Copy"
            style="display:inline-flex;align-items:center;justify-content:center;
                   width:auto;height:28px;padding:0;margin:0;
                   border:none;background:transparent;border-radius:6px;
                   cursor:pointer;color:#8a8a8a;font-size:16px;line-height:1;">
            <span id="{key}_icon">⧉</span> <span style="margin-left:4px;">copy</span>
        </button>
        <script>
            (function() {{
                var btn = document.getElementById("{key}");
                var icon = document.getElementById("{key}_icon");
                var text = {js_text};
                btn.addEventListener("mouseenter", function() {{ btn.style.background = "#efefef"; }});
                btn.addEventListener("mouseleave", function() {{ btn.style.background = "transparent"; }});
                btn.addEventListener("click", function() {{
                    navigator.clipboard.writeText(text);
                    icon.textContent = "✓";
                    icon.style.color = "#2ea043";
                    setTimeout(function() {{
                        icon.textContent = "⧉";
                        icon.style.color = "";
                    }}, 1500);
                }});
            }})();
        </script>
    """, unsafe_allow_javascript=True, width="content")


# --- Quick Search: clickable dropdown menu ---
with st.expander("🔍 Quick Search", expanded=len(st.session_state.messages) == 0):
    col1, col2 = st.columns(2)
    with col1:
        room_type = st.selectbox("Room type", ["Any", "1 in 1", "2 in 1", "3 in 1", "4 in 1"])
        max_price = st.selectbox("Max budget (GHS)", ["Any", 3000, 4000, 5000, 6000, 7000, 8000, 9000])
    with col2:
        max_distance = st.selectbox("Max walk to campus (min)", ["Any", 10, 15, 20, 30, 45, 60])
        amenities = st.multiselect("Amenities", ["Water", "WiFi", "Security", "TV room", "Study room"])

    if st.button("Search", type="primary", use_container_width=True):
        bits = []
        if room_type != "Any":
            bits.append(f"a {room_type} room")
        if max_price != "Any":
            bits.append(f"under GHS {max_price}")
        if max_distance != "Any":
            bits.append(f"within {max_distance} minutes of campus")
        if amenities:
            bits.append("with " + " and ".join(a.lower() for a in amenities))
        prompt = "Find me " + (", ".join(bits) if bits else "a hostel") + "."
        ask(prompt)
        st.rerun()

# Redraw all past messages on every rerun, with a copy button on each user message
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "user":
            copy_button(msg["content"], key=f"copy_{i}")

# Wait for new input
if prompt := st.chat_input("e.g. cheap 2-in-1 close to campus"):
    ask(prompt)
    st.rerun()