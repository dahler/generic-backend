
from flask import (
    Flask,
    json,
    jsonify,
    render_template,
    redirect,
    send_from_directory,
    session,
    url_for,
    request,
    flash,
)


from DBManager.DBManager import DBManager
from LLM.AIManager import AIManager
from DBManager import get_session

app = Flask(__name__)
# manager = AIManager("deepseek")
db_manager = DBManager(get_session)
ai_manager = AIManager("deepseek") 



@app.route("/api/chat-seaweed", methods=["POST"])
def chat_seaweed():
    return jsonify("Chat seaweed")


# @app.route('/api/ask', methods=['POST'])
# def ask_question():
#     data = request.json
#     question = data.get("question")
#     provider = data.get("provider", "openai").lower()

#     if not question:
#         return jsonify({"error": "Question is required"}), 400
    
#     answer = manager.check_what_to_search(question)

#     # if provider == "openai":
#     #     answer = ask_openai(question)
#     # elif provider == "deepseek":
#     #     answer = ask_deepseek(question)
#     # else:
#     #     return jsonify({"error": "Invalid provider. Use 'openai' or 'deepseek'"}), 400

#     print(answer)

#     return jsonify({"provider": provider, "question": question, "answer": answer})

@app.route('/api/ask', methods=['POST'])
async def ask_question():
    data = request.json
    question = data.get("question")
    provider = data.get("provider", "openai").lower()
    conversation_id = data.get("conversation_id")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    prev_con = None

    # Create or retrieve a conversation
    if conversation_id:
        conversation = db_manager.get_conversation(conversation_id)
        prev_con = db_manager.get_conversation_messages(conversation_id)
    else:
        # Create a new conversation
        print("Inserting new record")
        conversation = db_manager.create_conversation()
        conversation_id = conversation["id"]
        print(conversation_id)

    # Save the user's question to the conversation
    db_manager.save_message(content=question, role="user", conversation_id=conversation_id)

    # Generate the assistant's answer
    try:
        answer, error = await ai_manager.ask_ai(question, previous_conversation=prev_con, document_context=None)
        if error:
            print(error)
            return jsonify({"error": error}), 500

        # Save the assistant's answer to the conversation
        db_manager.save_message(content=answer, role="assistant", conversation_id=conversation_id)

        # Prepare the response
        response = {
            "conversation_id": conversation_id,
            "messages": answer,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)