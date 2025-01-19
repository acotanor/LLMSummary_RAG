# LLMSummary_RAG

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

The scope of this project is to avoid relying on third-party providers and costly subscriptions to use such usefull AI implementations as pdf summarization and RAGs. 

Any suggestion and/or improvements are welcomed!

---

# Table of contents:
* [Installation and setup:](#installation-and-setup)
    * [Set up the python environment](#1-setup-the-python-environment)
    * [Set up your local LLM](#2-setup-your-local-llm)
* [Summarizing pdf files:](#summarizing-pdf-files)
    * [Basic run](#basic-run)
    * [Advanced options](#advanced-options)
* [Building a RAG:](#building-a-rag)
    * [Preparing the data](#preparing-the-data)
    * [Asking questions about the provided data](#asking-questions-about-the-provided-data)
* [Customization:](#customization)
    * [Changing LLM](#changing-llm)
    * [HuggingFace Embeddings](#huggingface-embeddings)
* [Sources:](#sources)
    * [RAG](#rag)
    * [Summarizer](#summarizer)

---

# Installation and setup:
## 1. Setup the python environment:
You'll need to clone the repo in order to start working on it.

**Linux, Windows and MacOS:**
```bash
git clone https://github.com/acotanor/LLMSummary_RAG.git
```
```bash
cd LLMSummary_RAG
```
Once you've cloned the repo we can get started with the environment setup.

### Creating the environment:
**Linux and MacOS:**
```bash
python3 -m venv path/to/the/environment/.venv
```
**Windows:**
```bash
py -m venv \path\to\the\environment\.venv
```
### Activating the environment:
**Linux and MacOS:**
```bash
source /path/to/the/environment/.venv/bin/activate
```
**Windows:**
```bash
\path\to\the\environment\.venv\Scripts\activate
```
> [!IMPORTANT]
> As every .py file in the project requires imports you'll need to activate the environment every time in order for them to work.

### Prepare pip:

**Linux and MacOS:**
```bash
python3 -m pip install --upgrade pip
```
**Windows:**
```bash
py -m pip install --upgrade pip
```
### Install the requirements:
All the requirements are listed in the file requirements.txt, so the following command will install every requirement needed for the project. (It might take a while)

**Linux, Windows and MacOS:**
```bash
pip install -r requirements.txt
```

You can use this command to leave the virtual environment:
```bash
deactivate
```

More on python virtual environments:

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

## 2. Setup your local LLM:
There are many local LLMs to choose from but for simplicity's shake I'll explain how to setup Ollama's llama3. (I'll cover how to change the code to use different LLMs in the later sections.)

### How Ollama works:
Ollama consists of two main components: the client and the server. The client is a command line interface(CLI) for the user to interact with, and the server is a rest API wich runs on the port 11434 (it's easy to remember as llama=11434). So we need both components to use Ollama and its LLMs. 

The CLI sends POST and GET methods to the API, which then makes the LLMs interpret and execute or respond the query. The first step is handled by the python files in the repo, which we already have after cloning it, so we only need to deploy the API to start using the project.

### Download Ollama:

Download Ollama following their sites instructions:

https://ollama.com/download

### Download and start an LLM model with Ollama:

To download an LLM we first need to deploy the Ollama API.

**Linux and MacOS:**
```bash
ollama serve
```
**Windows:**

Ollama works a bit different on windows, instead of deploying the API with a command you need to open the Ollama app, which seems to do nothing but it actually starts the Ollama server on 11434. Then you can use the cmd just like a linux or macOS terminal.

Once the API is up run the following command to download the llama3 model:
```bash
ollama pull llama3
```
Thus finishing the setup process.

---

# Summarizing pdf files:

## Basic run:
This process is done by running llms.py, as simple as that.

Executing the following command will show how to use the program:
```bash
python3 llms.py -h
```
The basic syntax is as follows:
```bash
python3 llms.py --pdf "path/to/the/pdf/file" --sumary
```
I've written two basic prompts to get the best possible summaries in different styles, they're accessed by the flags --sumary and --key_points, which are required.

* --sumary: Summarizes the pdf explaining its core points.
* --key_points: Breaks down the important topics of the pdf and shows them in a bulleted list style.

Try running the next commands to see how both prompts work:
```bash
python3 llms.py --pdf "data/pdf/s41598-024-79048-0.pdf" --summary
python3 llms.py --pdf "data/pdf/s41598-024-79048-0.pdf" --key_points
```
The performance will be determined by your computers components, my pc takes a minute to summarize the example pdf, while my laptop needs to run llama2 and it still takes a few minutes to run.

## Advanced options:
### --chunk_size flag:
The flag --chunk_size (1024 by default) determines the size of the chunks of the extract() function. The extract function not only extracts the text, but, as it is usually to large to handle by the model, it also divides it into managable chunks of text which the model then treats separately and overlap to mantain the context. 1024 is the default because it is llama3's dimension, you can play with the value to see which one works best with your setup.
### Customizing the prompts:
If you need to tweak the prompts for your specific use just change the template variable (template is a PromptTemplate class variable) in either summarization_prompt or key_points_prompt variables definition.
### Tips:
For really big files (like books or long essays) the model may miss important information, so if you want to summarize a big file (80 pages+ more or less) I suggest to separate it by chapters/topics manually and then summarize each of them individually. It'll take longer but it'll work much better.

---

# Building a RAG:
## Preparing the data:
In order for the RAG to work it needs to prepare a database with the data before, which is done by populate_database.py.

populate_database.py automatically creates a folder called chroma, which stores the chroma database used by the model. And for that purpose I made a folder called data which contains two other folders: pdf and md.

### Creating/Updating the database:
The first step is to store the desired files on their respective folders, and then call populate_database.py to create the database with the provided files.

The performance will depend on your machine and ammount of data, don't worry if it takes a while.

The basic syntax is as follows:
```bash
python3 populate_database.py --pdf
python3 populate_database.py --md
```
Those commands automatically digest and store the data on all of the provided files for the rag model to use. As the example files cover completely different topics I suggest to only load one of them at a time, if you have different file types about similar topics (or a big dataset) it should work fine.

If you add new files to the data folders, running the command again updates it automatically, but if a file is removed you'll need to reset the database to remove the removed files data from the dataset.

### Reset de database
To reset the database use the --reset flag:
```bash
python3 populate_database.py --reset
```

## Asking questions about the provided data:
Once the database is ready we can start asking questions about the provided data, which is done by query_data.py.

The basic syntax is as follows:
```bash
python3 query_data.py "Question on the provided data"
```
That command will output the response to the question and a list the sources used to respond.

The flag --k changes the amount of top sources to generate the respond (the default is 5).
```bash
python3 query_data.py "Question on the provided data" --k 10
```
Will respond using the best 10 sources instead of the 5 best.

### Example use:
Try these commands to see an example of how the rag works:
```bash
python3 populate_database --pdf
python3 query_data.py "What is an LLM?"
python3 populate_database --reset
python3 populate_database --md
python3 query_data.py "How does Alice meet the Mad Hatter" --k 10
```
---

# Customization:
## Changing LLM:
There are two models that you can change and test, the embedding model and the language model. After a bit of testing I chose to use llama3 for both embeddings and text generation as it is quite fast and surprisingly light.

Changing either one only requires to change models.py's model class and the changes will apply to both the summaritation and RAG apps too.

Example: Changing the model from llama3 to llama2:
```python
# llama3 (default conf)
class model:
    def __init__(self, embeddings=OllamaEmbeddings(model="llama3"), llm="llama3"):
        ...
```
```python
# llama2 (custom conf)
class model:
    def __init__(self, embeddings=OllamaEmbeddings(model="llama2"), llm="llama2"):
        ...
```

## HuggingFace Embeddings:
HuggingFace models can be implemented to customize the embeddings, I experienced drops in performance while getting no apparent improvements in tex generation, but perhaps a more powerfull machine could benefit from a HuggingFace model bigger than llama3.

You'll need to do the following changes to models.py:
```python
import langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-m3" # The model that worked best for me.
model_kwargs = {"device":"cuda"} # Uses the gpu instead of the cpu.
encode_kwargs = {"normalize_embeddings":True}
hf_embeddings=HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
class model:
    def __init__(self, embeddings=hf_embeddings, llm="llama3"):
        ...
```
The model will be downloaded automatically when running the app for the first time.

Checkout the rest of HuggingFace embedding models and test which one suits you best:
https://huggingface.co/models?pipeline_tag=sentence-similarity

---

# Sources:
### Rag:
populate_database.py and query_data.py are modifications of  pixegami's work:
- Github repo: https://github.com/pixegami/rag-tutorial-v2
- Youtube channel: https://www.youtube.com/@pixegami
- Video in which he explains his code: https://www.youtube.com/watch?v=2TJxpyO3ei4

### Summarizer:
llms.py is based on debugverse's work:
- Github repo: https://github.com/debugverse/debugverse-youtube/tree/main/summarize_huge_documents_kmeans
- Youtube channel: https://www.youtube.com/@DebugVerseTutorials
- Video in which he explain his code: https://www.youtube.com/watch?v=Gn64NNr3bqU

Even though I've made a lot of changes I reccomend whatching their videos if you're interested in a more thorough explanation on how the code works.
