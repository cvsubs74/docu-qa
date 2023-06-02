from langchain.llms import AzureOpenAI
import os


class LLMOpenAI:
    def __init__(self, api_type, api_base, api_key):
        os.environ["OPENAI_API_TYPE"] = api_type
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key

    @staticmethod
    def load():
        return AzureOpenAI(temperature=0.9, deployment_name="text-davinci-003-dev1", model_name="text-davinci-003")
