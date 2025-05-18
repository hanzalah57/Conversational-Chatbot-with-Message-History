import streamlit as st
from dotenv import load_dotenv
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# -------------------------------
# Load API key from .env
# -------------------------------

st.set_page_config(
    page_title="🔧🤖Conversational Chatbot with Message History",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧🤖Conversational Chatbot with Message History")
st.sidebar.header("⚙️Configuration")
st.sidebar.header("🔑 Input Groq API Key")
api_key = st.sidebar.text_input("Please Enter your Groq API key:", type="password")  # API Key input

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue. ")
    st.stop()

# A placeholder to show success/failure message
if api_key:
    # Replace this block with actual API key validation if possible
    try:
        # Dummy validation logic; replace with real Groq API call to verify
        if api_key.startswith("gsk_") and len(api_key) > 10:
            st.sidebar.success("✅ API Key authentication successful!")
        else:
            st.sidebar.error("❌ Invalid API Key. Please try again.")
    except Exception as e:
        st.sidebar.error(f"Error validating API key: {e}")


# Sidebar Controls
model_name = st.sidebar.selectbox("Select Groq Model", 
    ["llama3-8b-8192","gemma2-9b-It","deepseek-r1-distill-llama-70b"])

temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.7  
)

max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 300, 150
)

# -------------------------------
# Memory Initialization
# -------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input Handling
# -------------------------------
user_input = st.chat_input("Ask Anything....")

if user_input:
    st.session_state.history.append(("user", user_input))

    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key
    )

    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    # Get AI response
    ai_response = conv.predict(input=user_input)

    # Append as tuple
    st.session_state.history.append(("assistant", ai_response))

# -------------------------------
# Chat Display
# -------------------------------
for role, text in st.session_state.history:
    if role == 'user':
        st.chat_message('user').write(text)
    else:
        st.chat_message('assistant').write(text)
