import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .Models import Base
from config.config import db_url

# Create the engine
# db_url = db_url
engine = create_engine(db_url)

# Create a configured "Session" class
Session = scoped_session(sessionmaker(bind=engine))

# Create all tables
Base.metadata.create_all(engine)

# Function to get a new session
def get_session():
    return Session()

# Optional: Cleanup session on exit
def cleanup():
    Session.remove()
