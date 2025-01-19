import argparse
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from models import model

# Args:
llms = argparse.ArgumentParser()
llms.add_argument("--pdf", type=str, help="Select the pdf file to summarize.", required=True)
llms.add_argument("--chunk_size", type=int, help="Select the size of the chunks when splitting the pdf file. (Default: 1024)", default=1024)

llms.add_argument("--llm_model", type=str, help="Select the model of the llm. (Default: llama3:latest)", default="llama3:latest")

promtp = llms.add_mutually_exclusive_group(required=True)
promtp.add_argument("--summary", action="store_true", help="The prompt will tell the model to summaryze the file.")
promtp.add_argument("--key_points", action="store_true", help="The prompt will tell the model to divide the file on its key points.")

args_llms = llms.parse_args()


# Default map prompt.
map_prompt = """
Write a summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

# Custom Prompts.
bullet_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
bullet_prompt_template = PromptTemplate(template=bullet_prompt, input_variables=["text"])

summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in detail. Break the summary into sections such as "
        "Introduction, Methods, Results, and Conclusion if applicable. Preserve technical details.\n\n"
        "Text:\n{text}"
    )
)

key_point_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Extract the key points from the following text and present them as a bulleted list. Ensure clarity and include all major ideas.\n\n"
        "Text:\n{text}"
    )
)

def extract(fpath): # Extracts the text and splits it into chunks.
    return RecursiveCharacterTextSplitter(chunk_size=args_llms.chunk_size, chunk_overlap=200).split_documents(PyPDFLoader(fpath).load())

def summarize_document(f, llm, embeddings): # Summarizes the text chunks.
    try:
        result = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=10).transform_documents(documents=extract(f))
        if args_llms.key_points:
            return load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=key_point_prompt).run(result)
        else:
            return load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=summarization_prompt).run(result)
    except Exception as e:
        return str(e)


llms_model = model(llm=args_llms.llm_model)
print(summarize_document(args_llms.pdf,llms_model.llm,llms_model.embeddings))