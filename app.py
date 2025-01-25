from flask import Flask, jsonify, request
from DBManager.DBManager import DBManager
from LLM.AIManager import AIManager
from DBManager import get_session
from KnowledgeManager.KnowledgeManager import KnowledgeManager

# Initialize Flask app
app = Flask(__name__)

# Initialize components
def initialize_app():
    print("starting Database ........")
    db_manager = DBManager(get_session)
    print(" ")

    print("starting AI Manager ........")
    ai_manager = AIManager("deepseek")
    print(" ")

    print("starting FAISS ........")
    know_manager = KnowledgeManager()
    print("")

    return db_manager, ai_manager, know_manager

# Global objects initialized once
db_manager, ai_manager, know_manager = initialize_app()

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
    """Generate a response using AI and save the assistant's answer."""
    try:
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

@app.route('/api/ask-indria-faiss', methods=['POST'])
async def ask_law_question():
    data = request.json
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conversation_id, prev_con = process_conversation(question, conversation_id)

    # Fetch context from knowledge manager
    ids = know_manager.search(question)
    context = " ".join(db_manager.get_context(id)["content"] for id in ids if db_manager.get_context(id))

    return await generate_response(question, conversation_id, prev_con, context)

@app.route('/api/ask-indria', methods=['POST'])
async def ask_law_question_bare():
    data = request.json
    question = data.get("question")
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conversation_id, prev_con = process_conversation(question, conversation_id)
    return await generate_response(question, conversation_id, prev_con)

# Run the app
if __name__ == "__main__":
    # Use a WSGI server like Gunicorn or Waitress in production
    # app.run(host="0.0.0.0", port=5000, debug=False)
    app.run(debug=False)
