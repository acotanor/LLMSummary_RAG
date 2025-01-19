from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

class model:
    def __init__(self, embeddings=OllamaEmbeddings(model="llama3"), llm="llama3"):
        # This setup allows to change only the embeddings and llm models name to be changed without invoking the whole method.
        self.embeddings = embeddings
        self.llm = OllamaLLM(model=llm)