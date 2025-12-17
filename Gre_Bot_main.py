import streamlit as st
from openai import OpenAI

# ==============================
# Configuration
# ==============================
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
FT_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:ageing-population-2025-12-16:CnRA8nHT"

st.set_page_config(page_title="Dual Model Comparison", layout="wide")

st.title("🔍 Dual Model Q&A Comparison")
st.markdown(
    "Ask a question and compare responses from the **Base Model** and the **Fine‑Tuned Model** side by side."
)

# ==============================
# API Key Handling
# ==============================
api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    help="Your key is used only for this session"
)

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# ==============================
# User Input
# ==============================
question = st.text_area(
    "Your Question",
    placeholder="e.g. What are the key challenges of an ageing population?",
    height=120
)

ask_button = st.button("Ask Both Models")

# ==============================
# Helper Function (Chat Completions – compatible)
# ==============================
def get_response(model_name: str, user_question: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_question},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==============================
# Run Inference
# ==============================
if ask_button and question.strip():
    with st.spinner("Querying models..."):
        base_answer = get_response(BASE_MODEL, question)
        ft_answer = get_response(FT_MODEL, question)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Base Model")
        st.caption(BASE_MODEL)
        st.write(base_answer)

    with col2:
        st.subheader("🎯 Fine‑Tuned Model")
        st.caption(FT_MODEL)
        st.write(ft_answer)

elif ask_button:
    st.warning("Please enter a question.")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Built with Streamlit + OpenAI Chat Completions API")