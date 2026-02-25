import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.globals import set_debug
from model import Model 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# set_debug(True)

web_paths = (
    "https://forbes.com.br/coluna/2026/02/bitcoin-em-queda-o-que-esta-por-tras-desse-movimento/",
    )

loader = WebBaseLoader(web_paths=web_paths)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200, 
                                               add_start_index=True
                                               )

splits = text_splitter.split_documents(docs)

oll_embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

input_test = "What is the main reason for the recent drop in Bitcoin's value according to the article?"
res = oll_embeddings.embed_query(input_test)

vector_store = Chroma.from_documents(documents=splits, embedding=oll_embeddings, collection_name="bitcoin_articles")

# Configurando o retriever para buscar os documentos mais relevantes
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Gerando os textos 
template_rag = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Pergunta: {pergunta}
    Contexto: {contexto}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

class_model = Model()

class_model.prompt =  PromptTemplate.from_template(template_rag)

format_docs = lambda docs: "\n\n".join([doc.page_content for doc in docs])

rag_chain =  ({"contexto": retriever | format_docs, 
               "pergunta": RunnablePassthrough()
               }
              | class_model.prompt
              | class_model.llm
              | StrOutputParser()
)

class_model.chain = rag_chain 

model_rag = class_model.chain

pergunta = "What is the main reason for the recent drop in Bitcoin's value according to the article?"

res = model_rag.invoke(pergunta)

print(res)

# print(len(res))

# print(len(splits))

# print(len(docs[0].page_content))