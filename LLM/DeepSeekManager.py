
from openai import OpenAI
from config.config import deepseek

class DeepSeekAIManager:

    # def __init__(self):


    async def ask_deepseek(self, message, model="deepseek-chat"):
        try:
            client = OpenAI(api_key=deepseek, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=model,
                temperature=0.9,
                stream=False
            )
            answer = response.choices[0].message.content
            return answer, None
        except Exception as e:
            return None, f"OpenAI Error: {str(e)}"
        except Exception as e:
            return None, f"An unexpected error occurred: {str(e)}"