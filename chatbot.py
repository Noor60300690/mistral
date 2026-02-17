import os
import streamlit as st
from mistralai import Mistral, UserMessage
from dotenv import load_dotenv

# ==============================
# Load API Key Securely
# ==============================
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

# ==============================
# Intent Classification Module
# ==============================
def classify_inquiry(inquiry):
    prompt = f"""
    You are a bank customer service bot.
    Categorize the inquiry into ONE of these categories:

    card arrival
    change pin
    exchange rate
    country support
    cancel transfer
    charge dispute
    customer service

    Only return the category name.

    Inquiry: {inquiry}
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )

    return response.choices[0].message.content.strip()


# ==============================
# Response Generation Module
# ==============================
def generate_response(inquiry, category):
    prompt = f"""
    You are a professional bank support assistant.

    The detected category is: {category}

    Provide a helpful, clear, and professional response
    to the following customer inquiry:

    {inquiry}

    Keep the tone friendly and concise.
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )

    return response.choices[0].message.content


# ==============================
# Summarization Module (Optional)
# ==============================
def summarize_text(text):
    prompt = f"""
    Summarize the following text clearly and concisely:

    {text}
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )

    return response.choices[0].message.content


# ==============================
# Streamlit Web Interface
# ==============================
st.set_page_config(page_title="AI Customer Support", page_icon="🏦")

st.title("AI Customer Support Chatbot")
st.write("Powered by Mistral AI")

option = st.radio("Choose mode:", ["Customer Support", "Summarize Text"])

user_input = st.text_area("Enter your message:")

if st.button("Submit") and user_input:

    if option == "Customer Support":
        category = classify_inquiry(user_input)
        answer = generate_response(user_input, category)

        st.subheader("Detected Category:")
        st.success(category)

        st.subheader("AI Response:")
        st.write(answer)

    elif option == "Summarize Text":
        summary = summarize_text(user_input)

        st.subheader("Summary:")
        st.write(summary)