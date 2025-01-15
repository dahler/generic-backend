import os
import openai
from LLM.DeepSeekManager import DeepSeekAIManager
from LLM.OpenAIManager import OpenAIManager


class AIManager:
    def __init__(self, provider="openai"):
        self.provider = provider
        base_dir = os.path.dirname(__file__)
        self.rules_general = self.load_rules(os.path.join(base_dir, "rules", "rules_general.txt"))
        self.rules_investment = self.load_rules(os.path.join(base_dir, "rules", "rules_investment.txt"))
        self.rules_usaha = self.load_rules(os.path.join(base_dir, "rules", "rules_usaha.txt"))
        self.rules_advise = self.load_rules(os.path.join(base_dir, "rules", "rules_advise.txt"))

    def load_rules(self, filepath):
        try:
            print(filepath)
            with open(filepath, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Rules file {filepath} not found.")
            return ""

    def set_provider(self, provider):
        """Set the AI service provider dynamically."""
        self.provider = provider

    def ask(self, prompt):
        """Route the question to the appropriate AI provider."""
        if self.provider == "openai":
            oi = OpenAIManager()
            return oi.ask_openai(prompt)
        elif self.provider == "deepseek":
            ds = DeepSeekAIManager()
            return ds.ask_deepseek(prompt)
        elif self.provider == "cloud":
            return self.ask_cloud(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        

    def check_what_to_search(self, question):
        prompt = f""" 
        {self.rules_general}

        Jawab Pertanyaan: {question}
        """
        return self.ask(prompt)
    
    def ask_ai(self, question, previous_conversation=None, document_context=None):
        context = ""
        if previous_conversation:
            context += f"Percakapan Sebelumnya: {previous_conversation}\n"
        if document_context:
            context += f"Data dari Dokumen: {document_context}\n"

        prompt = f"""
        {self.rules_general}

        {context}

        Pertanyaan: {question}
        Have the response in the format below
        Jawaban:
        """
        # print(prompt)
        return self.ask(prompt)

    def create_prompt_bank(self, question, previous_question=None, previous_conversation=None, document_context=None):
        context = ""
        if previous_conversation:
            context += f"Percakapan Sebelumnya: {previous_conversation}\n"
        if document_context:
            context += f"Data dari Dokumen: {document_context}\n"

        prompt = f"""
        {self.rules_general}

        {context}

        Pertanyaan: {question}
        Have the response in the format below
        Jawaban:
        Followup:
        """
        # print(prompt)
        return self.ask(prompt)

    def get_investment_return(self, question):
        prompt = f"""
        {self.rules_investment}

        Pertanyaan: {question}
        Asumsi:
        Perhitungan:
        Hasil:
        Saran:
        Followup:
        """

        # print (prompt)
        return self.ask(prompt)

    def get_usaha_return(self, question):
        prompt = f"""
        {self.rules_usaha}

        Pertanyaan: {question}
        Asumsi:
        Perhitungan:
        Hasil:
        Saran:
        """

        # print (prompt)
        return self.ask(prompt)
    
    def get_usaha_advise(self, question, context):
        prompt = f"""
        {self.rules_advise}
        user-context: {context}
        Pertanyaan: {question}
        """

        # print (prompt)
        return self.ask(prompt)
    
