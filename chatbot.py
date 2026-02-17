import os
import streamlit as st
from mistralai import Mistral, UserMessage
from dotenv import load_dotenv

# ==============================
# Load API Key Securely
# ==============================
load_dotenv()  # loads .env in same folder
api_key = os.getenv("MISTRAL_API_KEY")

st.set_page_config(page_title="AI Customer Support", page_icon="🏦")

# If API key is missing, stop early with a friendly message
if not api_key:
    st.error(
        "MISTRAL_API_KEY is missing.\n\n"
        "Fix:\n"
        "1) Create a file named `.env` in the SAME folder as this chatbot.py\n"
        "2) Add this line:\n"
        "   MISTRAL_API_KEY=your_key_here\n\n"
        "Then restart Streamlit."
    )
    st.stop()

# Create client only after confirming key exists
client = Mistral(api_key=api_key)

# Choose a safe default model (cheaper + usually available)
MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


# ==============================
# Helper: safe call to Mistral
# ==============================
def call_mistral(prompt: str) -> str:
    try:
        response = client.chat.complete(
            model=MODEL,
            messages=[UserMessage(content=prompt)],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Show readable message in UI instead of full traceback
        st.error(f"Error calling Mistral API: {e}")
        return ""


# ==============================
# Intent Classification Module
# ==============================
def classify_inquiry(inquiry: str) -> str:
    prompt = f"""
You are a bank customer service bot.
Categorize the inquiry into ONE of these categories exactly:

card arrival
change pin
exchange rate
country support
cancel transfer
charge dispute
customer service

Rules:
- Output ONLY the category name (no extra words, no punctuation).

Inquiry: {inquiry}
Category:
"""
    return call_mistral(prompt)


# ==============================
# Response Generation Module
# ==============================
def generate_response(inquiry: str, category: str) -> str:
    prompt = f"""
You are a professional bank support assistant.

Detected category: {category}

Write a helpful, clear, and professional response to the customer inquiry below.
Keep it friendly and concise (3-6 sentences max).

Customer inquiry:
{inquiry}
"""
    return call_mistral(prompt)


# ==============================
# Summarization Module
# ==============================
def summarize_text(text: str) -> str:
    prompt = f"""
Summarize the following text clearly and concisely in 3-5 bullet points:

{text}
"""
    return call_mistral(prompt)


# ==============================
# Streamlit Web Interface
# ==============================
st.title("AI Customer Support Chatbot")
st.caption(f"Powered by Mistral AI • Model: {MODEL}")

option = st.radio("Choose mode:", ["Customer Support", "Summarize Text"])
user_input = st.text_area("Enter your message:", height=150)

if st.button("Submit") and user_input.strip():

    if option == "Customer Support":
        with st.spinner("Classifying inquiry..."):
            category = classify_inquiry(user_input)

        if category:
            st.subheader("Detected Category:")
            st.success(category)

            with st.spinner("Generating response..."):
                answer = generate_response(user_input, category)

            if answer:
                st.subheader("AI Response:")
                st.write(answer)

    else:
        with st.spinner("Summarizing..."):
            summary = summarize_text(user_input)

        if summary:
            st.subheader("Summary:")
            st.write(summary)
