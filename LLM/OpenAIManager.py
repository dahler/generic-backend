
import openai
from config.config import openai_key

openai.api_key = openai_key

class OpenAIManager:

    # def __init__(self):


    def ask_openai(self, message, model="gpt-4o-mini"):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=0.9,
            )
            answer = response.choices[0].message.content
            return answer, None
        except openai.OpenAIError as e:
            return None, f"OpenAI Error: {str(e)}"
        except Exception as e:
            return None, f"An unexpected error occurred: {str(e)}"