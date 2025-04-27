import os
import bs4
import torch
from transformers import pipeline

# Evitiamo i warning USER_AGENT settando anche questo (facoltativo)
os.environ["USER_AGENT"] = "my-crawler/1.0"

# Impostazioni LangChain Studio (solo se vuoi tracciare)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

os.environ['LANGCHAIN_API_KEY'] = "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"
project_name="meglio"
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1) 
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-content","post-title","post-header")))
)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
splits = splitter.split_documents(docs)

# 2) 
embd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = Chroma.from_documents(splits, embedding=embd)
retriever = vs.as_retriever(search_kwargs={"k":2})

# 3) LLM su GPU
device = 0 if torch.cuda.is_available() else -1
gen = pipeline(
    "text-generation", model="gpt2", tokenizer="gpt2",
    max_new_tokens=128, do_sample=False, no_repeat_ngram_size=2,
    return_full_text=False, device=device
)
llm = HuggingFacePipeline(pipeline=gen)



system_prompt = (
    "You are a helpful assistant. Use the following context to answer the question. "
    "If you don't know the answer, say you don't know.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
combine_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# 5)
tracer = LangChainTracer(project_name=project_name)

# 6) 
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_chain
)

# 7) 
result = rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={"callbacks": [tracer]}
)