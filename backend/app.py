from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gtts import gTTS
from io import BytesIO
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import GROQ_API_KEY
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
store = {}


app = Flask(__name__, template_folder='templates')  # Specify folders
CORS(app, origins=["*"])

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
    #context = data.get("context")
    history = data.get("history", [])
    session_id = data.get("session_id")
    session_data = store.get(session_id)
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if not session_data:
        return jsonify({"error": "Session expired"}), 400

    index = session_data["index"]
    if index is None:
        return jsonify({"error": "PDF not processed yet"}), 400
    doc_chunks = session_data["chunks"]
    # Limit context (important)
    # Encode question
    query_embedding = model.encode([question]).astype("float32")

    # Search top 3 chunks
    D, I = index.search(query_embedding, k=3)

    retrieved_chunks = [doc_chunks[i] for i in I[0]]
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    sources = list(set([chunk["page"] for chunk in retrieved_chunks]))

    client = Groq(api_key=GROQ_API_KEY)
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
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

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
    store[session_id] = {
        "index":index,
        "chunks":doc_chunks
    }
    return jsonify({"status": "PDF processed"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
