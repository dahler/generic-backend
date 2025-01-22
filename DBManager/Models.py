import uuid
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UUID, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    created_at = Column(DateTime, default=datetime.utcnow)


    def to_dict(self):
        """Convert model instance to a dictionary."""
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

class Conversation(BaseModel):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(BaseModel):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(UUID, ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # e.g., 'user' or 'assistant'
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class LawDocument(Base):
    __tablename__ = "law_documents"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    metadata_column = Column("metadata", String(255))  # Alias the column name