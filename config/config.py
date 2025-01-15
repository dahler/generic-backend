import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# in local
APP_SECRET = os.getenv('APP_SECRET')
db_url = os.getenv('APP_DB')
openai_key = os.getenv('OPEN_AI')
deepseek = os.getenv('DEEP_SEEK_AI')


# in server
# APP_SECRET = os.environ.get('APP_SECRET')
# db_url = os.environ.get('APP_DB')