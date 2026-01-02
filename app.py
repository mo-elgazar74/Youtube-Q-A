# Youtube Q&A Project that takes the link from user and we go get the transcript from video metadata and if not available use base64 to extract the audio and transcribe it
# then we use langchain to generate the answer 
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
import re , os

HFTOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_video_id(youtube_url):
    # Extract the video ID from the YouTube URL using regex
    
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

def get_transcript(video_id):
    try:
        # Try to get the transcript using youtube-transcript-api
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["ar", "en"])   # بيرجع FetchedTranscript
        items = fetched.to_raw_data()
        # Return items with timestamps instead of just concatenated text
        return items
    except Exception as e:
        print(f"Transcript not available via API: {e}")
        # Fallback to audio extraction and transcription (not implemented here)
        return None
    
def main():
    youtube_url = input("Enter YouTube video URL: ")
    video_id = get_video_id(youtube_url)
    transcript_items = get_transcript(video_id)
    if not transcript_items:
        print("Transcript could not be retrieved.")
        return
    
    print("Processing transcript...")
    # Create documents with timestamps
    documents = []
    current_chunk = ""
    chunk_start_time = 0
    
    for i, item in enumerate(transcript_items):
        text = item['text']
        start_time = item['start']
        
        # Start a new chunk if empty
        if not current_chunk:
            chunk_start_time = start_time
            current_chunk = text
        # Add to current chunk if under 1000 chars
        elif len(current_chunk) + len(text) < 1000:
            current_chunk += " " + text
        # Otherwise, save current chunk and start new one
        else:
            # Convert timestamp to readable format (MM:SS)
            minutes = int(chunk_start_time // 60)
            seconds = int(chunk_start_time % 60)
            timestamp_str = f"{minutes}:{seconds:02d}"
            
            # Create YouTube link with timestamp
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
    
    # Add the last chunk
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
    print("Creating embeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        huggingfacehub_api_token=HFTOKEN
    )
    vector_store = FAISS.from_documents(documents, embeddings)
    
    
    # Create retriever and QA chain
    retriever = vector_store.as_retriever()
    
    # Initialize chat history store
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # Create prompt template that includes chat history
    template = """You are a helpful assistant answering questions about a YouTube video. Use the provided context from the video transcript to answer questions. Each piece of context includes a timestamp link.

Context from video:
{context}

Chat History:
{chat_history}

Current Question: {question}

Answer: Provide a detailed answer based on the context and include relevant timestamp links where this information can be found in the video."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the LLM
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    
    # Create the chain with timestamp formatting
    def format_docs(docs):
        formatted = []
        for doc in docs:
            timestamp = doc.metadata.get('timestamp', 'Unknown')
            timestamp_link = doc.metadata.get('timestamp_link', '')
            if timestamp_link:
                formatted.append(f"[{timestamp}] ({timestamp_link})\n{doc.page_content}")
            else:
                formatted.append(f"[{timestamp}] {doc.page_content}")
        return "\n\n".join(formatted)
    
    def format_chat_history(messages):
        """Format chat history, keeping only last 2 messages (1 Q&A pair)"""
        if len(messages) > 4:  # More than 2 Q&A pairs
            # Keep only the last 2 messages (last Q&A)
            messages = messages[-4:]
        
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        return "\n".join(formatted) if formatted else "No previous conversation."
    
    # Configuration for the session
    config = {"configurable": {"session_id": "1"}}
    
    print("\n" + "="*60)
    print("YouTube Q&A Chat - Type 'quit' or 'exit' to end the session")
    print("="*60 + "\n")
    
    # Conversation loop
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using YouTube Q&A Chat!")
            break
        
        if not question:
            continue
        
        print("Generating answer...")
        
        # Get session history
        session_history = get_session_history(config["configurable"]["session_id"])
        
        # Reformulate question with context if there's chat history
        search_query = question
        if len(session_history.messages) > 0:
            # Create a contextualized query using chat history
            chat_context = format_chat_history(session_history.messages)
            
            # Use LLM to reformulate the question with context
            reformulation_prompt = f"""Given the following conversation history and a new question, reformulate the question to be standalone and include relevant context from the conversation history.

Chat History:
{chat_context}

New Question: {question}

Reformulated Question (standalone, incorporating context):"""
            
            try:
                response = llm.invoke([HumanMessage(content=reformulation_prompt)])
                search_query = response.content.strip()
                if search_query:
                    print(f"[Searching for: {search_query}]")
                else:
                    search_query = question
            except:
                # If reformulation fails, use original question
                search_query = question
        
        # Retrieve relevant context using the reformulated query
        relevant_docs = retriever.invoke(search_query)
        context = format_docs(relevant_docs)
        
        # Format chat history
        chat_history_text = format_chat_history(session_history.messages)
        
        # Create the input for the chain
        chain_input = {
            "context": context,
            "chat_history": chat_history_text,
            "question": question
        }
        
        # Invoke the chain with streaming
        print("\nAnswer: ", end="", flush=True)
        answer = ""
        for chunk in (prompt | llm | StrOutputParser()).stream(chain_input):
            print(chunk, end="", flush=True)
            answer += chunk
        print()  # New line after streaming
        
        # Add to chat history
        session_history.add_user_message(question)
        session_history.add_ai_message(answer)

if __name__ == "__main__":
    main()