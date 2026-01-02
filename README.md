# 🎥 YouTube Q&A Chat

An AI-powered application that allows you to ask questions about any YouTube video and get intelligent answers with clickable timestamp links. Built with LangChain, Streamlit, and Groq's LLaMA model.

## 📖 About This Project

This project was developed after completing the **Foundation: Introduction to LangChain - Python** course from **LangChain Academy**, where I gained comprehensive knowledge about:

- Building AI applications with LangChain
- Working with vector stores and embeddings
- Implementing RAG (Retrieval-Augmented Generation)
- Creating conversational AI with memory
- Streaming responses for better UX

I applied these concepts to create a practical tool that makes YouTube videos more accessible and searchable through natural language queries.

## ✨ Features

- 🎯 **Smart Q&A**: Ask questions about any YouTube video in natural language
- ⚡ **Real-time Streaming**: Responses appear word-by-word as they're generated
- 🔗 **Timestamp Links**: Every answer includes clickable YouTube links to relevant moments
- 💬 **Chat History**: Maintains conversation context for follow-up questions
- 🌐 **Multi-language**: Supports both Arabic and English transcripts
- 🎨 **Beautiful UI**: Clean Streamlit interface for easy interaction
- 🚀 **Railway Ready**: Configured for one-click deployment

## 🛠️ Tech Stack

- **LangChain** - AI application framework
- **Groq** - Fast LLM inference (LLaMA 3.3 70B)
- **Streamlit** - Web UI framework
- **FAISS** - Vector similarity search
- **HuggingFace** - Embeddings (multilingual-e5-small)
- **YouTube Transcript API** - Transcript extraction

## 📦 Installation

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Setup

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd "LangChain Project"
```

2. **Install dependencies**

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
LANGSMITH_API_KEY=your_langsmith_key  # Optional
LANGSMITH_TRACING=true  # Optional
```

## 🚀 Usage

### Web Interface (Streamlit)

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

### Command Line Interface

```bash
python app.py
```

### How to Use

1. **Enter a YouTube URL** (e.g., `https://youtu.be/VIDEO_ID`)
2. **Wait for processing** - The app extracts and indexes the transcript
3. **Ask questions** - Type your questions in natural language
4. **Get answers** - Receive AI-generated responses with timestamp links
5. **Follow up** - Ask related questions; the app remembers context

## 🌐 Deployment to Railway

This project is configured for easy deployment to Railway:

1. **Push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Railway**

   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Add environment variables in Railway dashboard
   - Deploy automatically!

3. **Set Environment Variables** in Railway:
   - `GROQ_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`

## 📁 Project Structure

```
LangChain Project/
├── streamlit_app.py      # Streamlit web interface
├── app.py                # CLI version
├── pyproject.toml        # Dependencies
├── Procfile              # Railway deployment config
├── runtime.txt           # Python version
├── .env                  # Environment variables (local)
└── README.md             # This file
```

## 🎓 Key Concepts Applied

### RAG (Retrieval-Augmented Generation)

- Extracts YouTube transcripts and chunks them intelligently
- Creates vector embeddings for semantic search
- Retrieves relevant context before generating answers

### Conversational Memory

- Maintains chat history for context-aware responses
- Automatically trims old messages to prevent context overflow
- Reformulates vague questions using conversation history

### Streaming Responses

- Implements token-by-token streaming for better UX
- Shows real-time progress as the AI generates answers
- Provides immediate feedback to users

### Metadata Preservation

- Preserves timestamp information from transcripts
- Generates clickable YouTube links with `&t=` parameter
- Formats timestamps in human-readable format (MM:SS)

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **LangChain Academy** - Foundation: Introduction to LangChain - Python course
- **Groq** - For fast LLM inference
- **Streamlit** - For the amazing web framework
- **HuggingFace** - For embeddings models

## 📧 Contact

Created by Mohamed Elgazar - Feel free to reach out!

---

**Note**: This project was built as a practical application of concepts learned in the **Foundation: Introduction to LangChain - Python** course from **LangChain Academy**. It demonstrates real-world usage of RAG, conversational AI, and streaming responses.
