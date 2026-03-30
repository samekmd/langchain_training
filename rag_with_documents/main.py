import faiss
import tempfile
import os
import time 

import streamlit as st 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from agent_with_memory.main import model_hf_hub
from youtube_transcript.main import model_groq
from operator import itemgetter

import mlflow
from mlflow.genai.scorers import Correctness


from dotenv import load_dotenv
load_dotenv()

HUGGING_FACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

mlflow.langchain.autolog()

mlflow.set_experiment("RAG with documents")
mlflow.set_tracking_uri("http://localhost:5000")


# Indexação e recuperação de documentos usando FAISS
def config_retriever(uploads):
    # Carregar os documentos
    docs = []
    temp_dir = tempfile.TemporaryDirectory() 
    for file in uploads:
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_file_path)
        docs.extend(loader.load())

    # Dividir os documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Embeddings nomic-embed-text:latest | bge-m3:latest
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )
    
    # Armazenamento em FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("vector_store/db_faiss")
    
    # Configuração do retriever
    retriever = vectorstore.as_retriever(search_type="mmr",
                                         search_kwargs={'k':3, 'fetch_k': 4})
    
    return retriever
    
# Configuração da chain
def config_rag_chain(model_class, retriever):
    # Carregamento da llm 
    llm_dict = {
        "groq": model_groq(),
        "hf_hub": model_hf_hub()
    }
    
    llm = llm_dict[model_class]
    
    # Prompt de contextualização
    # Consulta -> retriever
    # (consulta, histórico do chat) -> LLM -> Consulta reformulada -> retriever
    context_q_system_prompt = """
       Your task is to REWRITE the user's query to be a standalone search term for a vector database.
RULES:
1. Analyze the chat history and the latest user message.
2. If the message refers to previous topics, reformulate it into a specific, independent question.
3. If it's already independent, return the original text exactly.
4. DO NOT answer the question.
5. DO NOT provide any conversational preamble (e.g., "Sure, here it is..." or "I suggest...").
6. Return ONLY the final reformulated text.
    """
    
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")   
        ]
    )
    
    query_chain = context_q_prompt | llm | StrOutputParser() 
    
    # Chain para contextualização da consulta
    history_aware_retriever = RunnableBranch(
        (
            lambda x: bool(x.get("chat_history")),
            query_chain | retriever,
        ),
        itemgetter("input") | retriever
    )
 
 
    # Prompt de perguntas e respostas
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    
    full_chain = RunnablePassthrough.assign(context=history_aware_retriever) | {
        "answer": qa_prompt | llm | StrOutputParser(),
        "context": itemgetter("context")
    }
    
    with mlflow.start_run(run_name="RAG Chain Configuration"):
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("retriever_type", "FAISS")
        mlflow.log_param("embedding_model", "nomic-embed-text:latest") 
    
    return full_chain

if __name__ == "__main__":
    # Configuração do streamlit, with emojis
    st.set_page_config(page_title="Converse com documentos", page_icon="📖")
    st.title("Converse com documentos 📖")

    # Criação de painel lateral para upload de arquivos
    uploads = st.sidebar.file_uploader(
        label="Enviar arquivos", 
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if not uploads:
        st.info("Por favor envie algum arquivo para continuar")
        st.stop()
        
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage(content="Olá! Como posso ajudar você hoje?")]
        
    if "docs_list" not in st.session_state:
        st.session_state["docs_list"] = None
        
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                
    start = time.time()
    user_query = st.chat_input("Digite sua mensagem")
    
    if user_query and uploads:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            if st.session_state.docs_list != uploads:
                st.session_state.docs_list = uploads
                st.session_state.retriever = config_retriever(uploads)
                
            rag_chain = config_rag_chain(model_class='groq', retriever=st.session_state.retriever)
            
            result = rag_chain.invoke({"input":user_query, "chat_history": st.session_state.chat_history})
            
            resp = result['answer']
            st.write(resp)
            
            # Mostrar a fonte
            sources = result['context']
            for idx, doc in enumerate(sources):
                source = doc.metadata.get('source', 'Desconhecido')
                file = os.path.basename(source)
                page = doc.metadata.get('page', 'N/A')
                
                with st.popover(f"Fonte {idx+1}: {file}"):
                    st.caption(f"Página: {page}")
                    st.write(doc.page_content)
              
                    
        st.session_state.chat_history.append(AIMessage(content=resp))
    end = time.time()
    print(f'Tempo: {end - start}')