import os
import asyncio
from flask import Flask, jsonify, request
from DBManager.DBManager import DBManager
from DBManager.Models import FinanceDocument, LawDocument
from LLM.AIManager import AIManager
from DBManager import get_session
from KnowledgeManager.KnowledgeManager import KnowledgeManager
from KnowledgeManager.BM25Manager import BM25Manager
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize components
print("Starting Database ........")
db_manager = DBManager(get_session)
print(" ")

print("Starting AI Manager ........")
ai_manager = AIManager("deepseek")
print(" ")

print("Starting FAISS ........")
law_manager = KnowledgeManager("faiss_index")
duit_manager = KnowledgeManager("faiss_finance_index")
print(" ")

bmManager = BM25Manager()

def process_conversation(question: str, conversation_id: str = None):
    """Handle conversation creation or retrieval and save the user's question."""
    prev_con = None

    if conversation_id:
        conversation = db_manager.get_conversation(conversation_id)
        prev_con = db_manager.get_conversation_messages(conversation_id)
    else:
        conversation = db_manager.create_conversation()
        conversation_id = conversation["id"]

    db_manager.save_message(content=question, role="user", conversation_id=conversation_id)
    return conversation_id, prev_con

async def generate_response(question: str, conversation_id: str, prev_con, context=None):
    """Generate a response using AI asynchronously but keep Flask synchronous."""
    try:
        # Directly await the AI function
        answer, error = await ai_manager.ask_law_ai(question, previous_conversation=prev_con, document_context=context)

        if error:
            return jsonify({"error": error}), 500

        db_manager.save_message(content=answer, role="assistant", conversation_id=conversation_id)
        return jsonify({"conversation_id": conversation_id, "messages": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat-seaweed", methods=["POST"])
def chat_seaweed():
    return jsonify("Chat seaweed")

@app.route('/api/ask-indria', methods=['POST'])
async def ask_law_question():
    data = request.get_json()
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conversation_id, prev_con = process_conversation(question, conversation_id)

    # Fetch context from knowledge manager (awaiting the search method)
    faiss_results  = await law_manager.search(question)

    # print(ids)
    # context_results = [db_manager.get_context(id, LawDocument) for id in ids]

    retrieved_docs = []
    for doc_id, _ in faiss_results:
        doc = db_manager.get_context(doc_id, LawDocument)
        if doc:
            retrieved_docs.append({"id": doc_id, "text": doc["content"]})
    
    if not retrieved_docs:
        print(conversation_id)
        return await generate_response(question, conversation_id, prev_con, "")

    doc_texts = [doc["text"] for doc in retrieved_docs]

    bm25_scores = bmManager.getBM25Score(question, doc_texts)
    print(bm25_scores)
    # Normalize FAISS distances (invert since FAISS uses distances)
    faiss_scores = np.array([1 - (dist / max(faiss_results, key=lambda x: x[1])[1]) for _, dist in faiss_results])

    # Normalize BM25 Scores
    scaler = MinMaxScaler()
    bm25_scores = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
    faiss_scores = scaler.fit_transform(faiss_scores.reshape(-1, 1)).flatten()

    # Hybrid Score Calculation
    alpha = 0.6  # Weight for BM25
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # Rank and Select Best Context
    ranked_results = sorted(zip(retrieved_docs, hybrid_scores), key=lambda x: x[1], reverse=True)

    print("New ranking")
    print([doc["id"] for doc, _ in ranked_results[:3]])  # Print top 3 document indices

    best_context = " ".join([doc["text"] for doc, _ in ranked_results[:3]])  # Top 3 documents

    return await generate_response(question, conversation_id, prev_con, best_context)

@app.route('/api/ask-duit', methods=['POST'])
async def ask_duit_question():
    data = request.get_json()
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conversation_id, prev_con = process_conversation(question, conversation_id)

    faiss_results = await duit_manager.search(question)

    # context_results = [db_manager.get_context(id, FinanceDocument) for id in faiss_results]
    # # print(context_results)

    # context = " ".join(res["content"] for res in context_results if res)

    # return await generate_response(question, conversation_id, prev_con, context)
    retrieved_docs = []
    for doc_id, _ in faiss_results:
        doc = db_manager.get_context(doc_id, FinanceDocument)
        if doc:
            retrieved_docs.append({"id": doc_id, "text": doc["content"]})
    
    if not retrieved_docs:
        print(conversation_id)
        return await generate_response(question, conversation_id, prev_con, "")

    doc_texts = [doc["text"] for doc in retrieved_docs]

    bm25_scores = bmManager.getBM25Score(question, doc_texts)
    print(bm25_scores)
    # Normalize FAISS distances (invert since FAISS uses distances)
    faiss_scores = np.array([1 - (dist / max(faiss_results, key=lambda x: x[1])[1]) for _, dist in faiss_results])

    # Normalize BM25 Scores
    scaler = MinMaxScaler()
    bm25_scores = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
    faiss_scores = scaler.fit_transform(faiss_scores.reshape(-1, 1)).flatten()

    # Hybrid Score Calculation
    alpha = 0.6  # Weight for BM25
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # Rank and Select Best Context
    ranked_results = sorted(zip(retrieved_docs, hybrid_scores), key=lambda x: x[1], reverse=True)

    print("New ranking")
    print([doc["id"] for doc, _ in ranked_results[:3]])  # Print top 3 document indices

    best_context = " ".join([doc["text"] for doc, _ in ranked_results[:3]])  # Top 3 documents

    return await generate_response(question, conversation_id, prev_con, best_context)

# Run the app
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
