# LLMSummary_RAG

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

The scope of this project is to avoid relying on third-party providers and costly subscriptions to use such usefull AI implementations as pdf summarization and RAGs. 

Any suggestion and/or improvements are welcomed!

---

# Table of contents:
* [Installation and setup:](#installation-and-setup)
    * [Set up the python environment](#1-setup-the-python-environment)
    * [Set up your local LLM](#2-setup-your-local-llm)
* [Summarizing pdf files.](#summarizing-pdf-files)
* [Building a RAG:](#building-a-rag)
    * [Preparing the data](#preparing-the-data)
    * [Asking questions about the provided data](#asking-questions-about-the-provided-data)
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
All the requirements are listed in the file requirements.txt, so the following command will install every requirement needed for the project.

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

### Download Ollama

---

# Summarizing pdf files:



---

# Building a RAG:

## Preparing the data:

## Asking questions about the provided data:

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
