from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gtts import gTTS
from io import BytesIO
from groq import Groq
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

store = {}


app = Flask(__name__, template_folder='templates')  # Specify folders
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://documind-ai-iota.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response
CORS(
    app,
    supports_credentials=True,
    origins=[
        "https://documind-ai-iota.vercel.app"
    ]
)

@app.route('/', methods=['GET'])
def checking():
    print("documind api is running")
    
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "session_id": "ok"}), 200

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
    data = request.get_json()

    question = data.get("question")
    history = data.get("history", [])
    session_id = data.get("session_id")

    session_data = store.get(session_id)

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    if not session_data:
        return jsonify({"error": "Session expired. Please upload PDF again."}), 400

    embeddings = session_data["embeddings"]
    doc_chunks = session_data["chunks"]
    vectorizer = session_data["vectorizer"]

    # ✅ Convert question to vector
    query_vec = vectorizer.transform([question])

    # ✅ Compute similarity
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = scores.argsort()[-3:][::-1]
    retrieved_chunks = [doc_chunks[i] for i in top_indices]
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    sources = list(set([chunk["page"] for chunk in retrieved_chunks]))

    client = Groq(api_key=GROQ_API_KEY)
    

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant helping a user understand a PDF. Use provided context and conversation history."
            }
        ]

        # Add history
        for msg in history:
            messages.append(msg)

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

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    print("STEP 1: request received")

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    text = data.get("text", "")
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    if not text.strip():
        return jsonify({"error": "Text content is empty"}), 400

    chunk_size = 300
    doc_chunks = []

    pages = text.split("\n")

    for page_num, page in enumerate(pages, start=1):
        words = page.split()

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            doc_chunks.append({
                "text": chunk_text,
                "page": page_num
            })

    # ✅ TF-IDF embeddings
    texts = [chunk["text"] for chunk in doc_chunks]

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(texts)

    store[session_id] = {
        "embeddings": embeddings,
        "chunks": doc_chunks,
        "vectorizer": vectorizer
    }

    return jsonify({"status": "PDF processed", "chunks": len(doc_chunks)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
