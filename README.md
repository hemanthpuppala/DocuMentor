# DocuMentor

**AI-powered research assistant** with multi-agent processing. Ask questions, attach documents, and get structured research reports backed by web search and academic papers (arXiv). Includes follow-up Q&A, chat, and auto-generated mock tests.

![DocuMentor](https://github.com/hemanthpuppala/DocuMentor/raw/main/clean_professional_workflow.png)

## Features

- **Research pipeline**: Upload PDFs/docs → query enhancement → web search (Brave) + arXiv → source summarization → structured report
- **Follow-up Q&A**: Ask follow-up questions in context of your research and chat history
- **Mock tests**: Generate 10-question quizzes from research content and get AI-graded feedback
- **Sessions**: Multiple research sessions with persistent chat and summaries

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain (Google Gemini / Groq)
- **Frontend**: Static HTML/CSS/JS served by FastAPI
- **Document processing**: PyMuPDF (PDF), python-docx (DOCX), plain text

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/hemanthpuppala/DocuMentor.git
cd DocuMentor
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment

Copy the example env and add your keys:

```bash
cp .env.example .env
# Edit .env and set at least GOOGLE_API_KEY or GROQ_API_KEY
```

### 3. Run

```bash
python main.py
```

- **App**: http://127.0.0.1:8000  
- **API docs**: http://127.0.0.1:8000/docs  
- **Health**: http://127.0.0.1:8000/health  

### Alternative: enhanced main

For follow-up routing, enhanced DOCX handling, and research vs chat follow-ups:

```bash
python enhanced_main.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | One of these | Google Gemini (primary LLM) |
| `GROQ_API_KEY` | One of these | Groq fallback LLM |
| `BRAVE_API_KEY` | Optional | Web search (Brave Search API) |

See [.env.example](.env.example) for the full list.

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend |
| `/query` | POST | Submit research question (+ optional files) |
| `/followup` | POST | Research-style follow-up (same session) |
| `/followup-chat` | POST | Chat-style follow-up |
| `/mock-test/{session_id}` | POST | Generate quiz for session |
| `/mock-test/{session_id}/submit` | POST | Submit quiz answers, get grading report |
| `/sessions` | GET | List sessions |
| `/session/{id}` | GET | Get session summary |
| `/session/{id}/full` | GET | Full session + chat history |
| `/new-session` | POST | Create new session |
| `/health` | GET | Service health |

## Project Structure

```
DocuMentor/
├── main.py              # Standard backend (FastAPI + LangGraph)
├── enhanced_main.py     # Enhanced backend (follow-up routing, DOCX)
├── rag_enhanced_main.py # RAG-enhanced variant
├── static/
│   ├── index.html       # Frontend
│   ├── script.js
│   └── styles.css
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT.
