import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .Models import Base
from config.config import db_url

# print(db_url)
engine = create_engine(db_url,
                    pool_size=5,  # Number of connections in the pool
                    max_overflow=10,  # Extra connections beyond the pool size
                    pool_timeout=30,  # Seconds to wait before timing out   
                       )

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
