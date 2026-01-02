import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import re
import os

# Page config
st.set_page_config(
    page_title="YouTube Q&A Chat",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #FF0000;
    }
</style>
""", unsafe_allow_html=True)

HFTOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_video_id(youtube_url):
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["ar", "en"])
        items = fetched.to_raw_data()
        return items
    except Exception as e:
        st.error(f"Transcript not available: {e}")
        return None

@st.cache_resource
def process_video(youtube_url):
    """Process video and create vector store - cached for performance"""
    video_id = get_video_id(youtube_url)
    transcript_items = get_transcript(video_id)
    
    if not transcript_items:
        return None, None
    
    # Create documents with timestamps
    documents = []
    current_chunk = ""
    chunk_start_time = 0
    
    for item in transcript_items:
        text = item['text']
        start_time = item['start']
        
        if not current_chunk:
            chunk_start_time = start_time
            current_chunk = text
        elif len(current_chunk) + len(text) < 1000:
            current_chunk += " " + text
        else:
            minutes = int(chunk_start_time // 60)
            seconds = int(chunk_start_time % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"
            timestamp_link = f"https://www.youtube.com/watch?v={video_id}&t={int(chunk_start_time)}"
            
            documents.append(Document(
                page_content=current_chunk,
                metadata={
                    "timestamp": timestamp_str,
                    "start_seconds": chunk_start_time,
                    "timestamp_link": timestamp_link
                }
            ))
            current_chunk = text
            chunk_start_time = start_time
    
    if current_chunk:
        minutes = int(chunk_start_time // 60)
        seconds = int(chunk_start_time % 60)
        timestamp_str = f"{minutes}:{seconds:02d}"
        timestamp_link = f"https://www.youtube.com/watch?v={video_id}&t={int(chunk_start_time)}"
        documents.append(Document(
            page_content=current_chunk,
            metadata={
                "timestamp": timestamp_str,
                "start_seconds": chunk_start_time,
                "timestamp_link": timestamp_link
            }
        ))
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        huggingfacehub_api_token=HFTOKEN
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store, video_id

def format_docs(docs):
    formatted = []
    for doc in docs:
        timestamp = doc.metadata.get('timestamp', 'Unknown')
        timestamp_link = doc.metadata.get('timestamp_link', '')
        if timestamp_link:
            formatted.append(f"[{timestamp}]({timestamp_link})\n{doc.page_content}")
        else:
            formatted.append(f"[{timestamp}] {doc.page_content}")
    return "\n\n".join(formatted)

def format_chat_history(messages):
    if len(messages) > 4:
        messages = messages[-4:]
    
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted) if formatted else "No previous conversation."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Header
st.title("🎥 YouTube Q&A Chat")
st.markdown("Ask questions about any YouTube video with AI-powered answers and timestamp links!")

# Sidebar for video input
with st.sidebar:
    st.header("📹 Video Setup")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtu.be/...",
        help="Enter a YouTube video URL to analyze"
    )
    
    if st.button("Process Video", type="primary"):
        if youtube_url:
            with st.spinner("Processing video transcript..."):
                try:
                    vector_store, video_id = process_video(youtube_url)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.video_id = video_id
                        st.session_state.messages = []
                        st.session_state.chat_history = InMemoryChatMessageHistory()
                        st.success("✅ Video processed successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error processing video: {e}")
        else:
            st.warning("Please enter a YouTube URL")
    
    if st.session_state.video_id:
        st.success(f"✅ Video loaded: {st.session_state.video_id}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = InMemoryChatMessageHistory()
            st.rerun()

# Main chat interface
if st.session_state.vector_store is None:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Setup LLM and retriever
            llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
            retriever = st.session_state.vector_store.as_retriever()
            
            # Retrieve context
            relevant_docs = retriever.invoke(prompt)
            context = format_docs(relevant_docs)
            chat_history_text = format_chat_history(st.session_state.chat_history.messages)
            
            # Create prompt
            template = """You are a helpful assistant answering questions about a YouTube video. Use the provided context from the video transcript to answer questions. Each piece of context includes a timestamp link.

Context from video:
{context}

Chat History:
{chat_history}

Current Question: {question}

Answer: Provide a detailed answer based on the context and include relevant timestamp links where this information can be found in the video."""
            
            prompt_template = ChatPromptTemplate.from_template(template)
            
            chain_input = {
                "context": context,
                "chat_history": chat_history_text,
                "question": prompt
            }
            
            # Stream response
            full_response = ""
            for chunk in (prompt_template | llm | StrOutputParser()).stream(chain_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.add_user_message(prompt)
        st.session_state.chat_history.add_ai_message(full_response)
