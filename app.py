from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from gtts import gTTS
from io import BytesIO
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import os

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_chunks = []
doc_embeddings = None
index = None
app = Flask(__name__, template_folder='templates')  # Specify folders
CORS(app)
@app.route('/')
def home():
    return render_template("index.html")
    
@app.route('/favicon.ico')
def favicon():
    from flask import send_from_directory
    return send_from_directory('templates', 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
@app.route('/tts', methods=['POST'])
def tts():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text.strip():
            return jsonify({"error": "Text is empty"}), 400

        tts = gTTS(text, lang='en')
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)

        return send_file(
            audio_stream,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='speech.mp3'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global index, doc_chunks
    data = request.get_json()
    question = data.get("question")
    context = data.get("context")
    history = data.get("history", [])

    # Limit context (important)
    # Encode question
    query_embedding = model.encode([question]).astype("float32")

    # Search top 3 chunks
    D, I = index.search(query_embedding, k=3)

    retrieved_chunks = [doc_chunks[i] for i in I[0]]
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    sources = list(set([chunk["page"] for chunk in retrieved_chunks]))

    client = Groq(api_key="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")
    prompt = f"""
        You are an AI assistant helping a user understand a PDF page.

        Use ONLY the provided context.

        Instructions:
        - If user asks for summary → give concise bullet summary
        - If user asks for explanation → explain simply
        - If user asks specific question → answer directly
        - If answer not in context → say "Not found in this page"

        Keep answers clear and structured.

        Context:
        {context}

        User Question:
        {question}
        """

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant helping a user understand a PDF. Use provided context and conversation history."
            }
        ]

        # Add history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add latest question with context
        messages.append({
            "role": "user",
            "content": f"""
    Context:
    {context}

    Question:
    {question}
    """
        })

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        answer = response.choices[0].message.content

        return jsonify({"answer": answer,"sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global doc_chunks, doc_embeddings, index

    data = request.get_json()
    text = data.get("text", "")

    # 🔹 Step 1: chunking
    chunk_size = 300
    words = text.split()

    doc_chunks = []
    chunk_size = 300

    pages = text.split("\n")  # each page separated

    for page_num, page in enumerate(pages, start=1):
        words = page.split()

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])

            doc_chunks.append({
                "text": chunk_text,
                "page": page_num
            })

    # 🔹 Step 2: embeddings
    embeddings = model.encode([chunk["text"] for chunk in doc_chunks])

    doc_embeddings = np.array(embeddings).astype("float32")

    # 🔹 Step 3: FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    return jsonify({"status": "PDF processed"})

if __name__ == '__main__':
    app.run()
