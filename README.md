# Vocab Cards

Generate printable vocabulary flashcard PDFs with AI-generated images.

## Setup

```bash
cd vocab-cards
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000
