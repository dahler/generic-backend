from .Models import Conversation, LawDocument, Message
from sqlalchemy.orm import scoped_session


class DBManager:
    def __init__(self, session_factory: scoped_session):
        self.session_factory = session_factory

    def create_conversation(self):
        """Create a new conversation and persist it."""
        with self.session_factory() as session:
            conversation = Conversation()
            session.add(conversation)
            session.commit()
            return conversation.to_dict()

    def get_conversation(self, conversation_id):
        """Retrieve a conversation by ID."""
        with self.session_factory() as session:
            conversation = session.query(Conversation).get(conversation_id)
            return conversation

    def save_message(self,  content, role, conversation_id):
        """Save a new message to a conversation."""
        with self.session_factory() as session:
            message = Message(content=content, role=role,  conversation_id=conversation_id)
            session.add(message)
            session.commit()
            return message.to_dict()

    def get_conversation_messages(self, conversation_id):
        """Retrieve all messages in a conversation."""
        with self.session_factory() as session:
           messages = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)  # Optional: ensure messages are in chronological order
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def get_context(self, id):
        """Retrieve a single document by its ID."""
        with self.session_factory() as session:
            # Query the database
            document = (
                session.query(LawDocument.content, LawDocument.metadata_column)
                .filter(LawDocument.id == int(id))
                .first()
            )
        
            # Handle the case where no document is found
            if not document:
                return None

            # Construct and return the result
            return {
                "content": document.content,
                "metadata": document.metadata_column
            }

            

