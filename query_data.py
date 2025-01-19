import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from models import model

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer to the question based only on the above context, preserve technical details and do it in two paragraphs: {question}
"""
query = argparse.ArgumentParser()
query.add_argument("query_text", type=str, help="The query text.")
query.add_argument("--llm_model", type=str, help="Select the model of the llm. (Default: llama3:latest)", default="llama3:latest")
args_query = query.parse_args()

model_query = model(llm=args_query.llm_model)

def main():
    query_rag(args_query.query_text)

def query_rag(query_text: str):
    # Search the DB.
    results = Chroma(persist_directory=CHROMA_PATH, embedding_function=model_query.embeddings).similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    response_text = model_query.llm.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()