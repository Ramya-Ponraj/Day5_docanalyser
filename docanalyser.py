import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from docx import Document

# ------------------- CONFIG -------------------
# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUqoyAAYv4Veg3hwrfELljrC_-MkcJMpg"

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize chat history list for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- FUNCTIONS -------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="📊 RAG-Based Data Analyzer", layout="centered")
st.title("📄 RAG-based Document Analyzer with Gemini")

# Custom CSS for WhatsApp-like chat styling
st.markdown("""
<style>
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 10px;
    background-color: #f5f5f5;
}
.user-message {
    background-color: #d1ffd1;
    padding: 10px;
    margin: 5px 10px;
    border-radius: 10px;
    max-width: 70%;
    margin-left: auto;
    text-align: left;
}
.ai-message {
    background-color: #ffffff;
    padding: 10px;
    margin: 5px 10px;
    border-radius: 10px;
    max-width: 70%;
    margin-right: auto;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

document_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document_text = extract_text_from_docx(uploaded_file)
    st.success("✅ Document successfully uploaded and processed!")

# Chat section
if document_text:
    st.markdown("---")
    st.subheader("💬 Chat with the Document")

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, (role, content) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-message">{content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Question input
    question = st.text_input("Your question:", key="question_input")

    if st.button("Send"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                # Add user question to chat history
                st.session_state.chat_history.append(("user", question))

                # Save user question to memory
                st.session_state.memory.save_context({"input": question}, {"output": ""})

                # Prompt with conversation history
                prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant. Use the following document context and conversation history to answer the user's question.\n\n"
                    "Context:\n{context}\n\n"
                    "Conversation History:\n{history}\n\n"
                    "Question:\n{question}"
                )

                # Prepare the chain
                chain: Runnable = prompt | llm

                # Get conversation history
                history = st.session_state.memory.load_memory_variables({})["history"]

                # Get AI response
                response = chain.invoke({"context": document_text, "question": question, "history": history})
                answer = response.content if hasattr(response, "content") else str(response)

                # Save AI response to memory and chat history
                st.session_state.memory.save_context({"input": question}, {"output": answer})
                st.session_state.chat_history.append(("ai", answer))

                st.success("✅ Answer generated successfully!")
                # Rerun to refresh chat history display
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error while generating answer: {e}")