import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agent_with_memory.main import model_hf_hub
from youtube_transcript.main import model_groq
from rag_with_documents.main import config_retriever
from query_translation_prompts import QueryTranslationPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

import mlflow

mlflow.langchain.autolog()

mlflow.set_experiment("RAG with documents Step Back")
mlflow.set_tracking_uri("http://localhost:5000")

from dotenv import load_dotenv
load_dotenv()

HUGGING_FACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def config_llm(model_class,):
    # Carregamento da llm 
    llm_dict = {
        "groq": model_groq(),
        "hf_hub": model_hf_hub()
    }
    
    llm = llm_dict[model_class]
    
    return llm
    

def multy_query_retrieval_chain(llm, retriever):
    query_translation_prompt = QueryTranslationPromptTemplate(llm, retriever)
    
    retrieval_chain = query_translation_prompt.multi_query()
    
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain = (
        {"context": retrieval_chain,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    with mlflow.start_run(run_name="RAG Chain Configuration"):
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("retriever_type", "FAISS")
        mlflow.log_param("embedding_model", "nomic-embed-text:latest") 
    
    return final_rag_chain


def multy_query_retrieval_chain_rag_fusion(llm, retriever):
    query_translation_prompt = QueryTranslationPromptTemplate(llm, retriever)
    
    retrieval_chain_rag_fusion = query_translation_prompt.rag_fusion()
    
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    final_rag_chain_rag_fusion = (
        {"context": retrieval_chain_rag_fusion,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    with mlflow.start_run(run_name="RAG Chain Configuration - RAG Fusion"):
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("retriever_type", "FAISS")
        mlflow.log_param("embedding_model", "nomic-embed-text:latest") 
    
    return final_rag_chain_rag_fusion


def decomposition_chain(llm, retriever, question):
    query_translation_prompt = QueryTranslationPromptTemplate(llm, retriever)
    
    questions = query_translation_prompt.decomposition(question)
    
    template = """Here is the question you need to answer:
                \n --- \n {question} \n --- \n
                Here is any available background question + answer pairs:
                \n --- \n {q_a_pairs} \n --- \n
                Here is additional context relevant to the question: 
                \n --- \n {context} \n --- \n
                Use the above context and any background question + answer pairs to answer the question: \n {question}
                """
    
    decomposition_prompt = ChatPromptTemplate.from_template(template)
    answer = ""
    q_a_pairs = ""
    
    
    with mlflow.start_run(run_name="RAG Chain Configuration - Decomposition"):
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("retriever_type", "FAISS")
        mlflow.log_param("embedding_model", "nomic-embed-text:latest") 
    
        for q in questions:
            rag_chain = (
                {"context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs")}
                | decomposition_prompt
                | llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pair = query_translation_prompt.format_qa_pair(q, answer)
            q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
    
    return answer



def step_back(llm, retriever, question):
    query_translation_prompt = QueryTranslationPromptTemplate(llm, retriever)
    
    generate_queries_step_back = query_translation_prompt.step_back()
    
    response_prompt_template = """You are an expert of business world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
                                # {normal_context}
                                # {step_back_context}

                                # Original Question: {question}
                                # Answer:""" 
                                
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
        {
            # Retrieve context using the original question
            "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": RunnableLambda(lambda x: x['question']) 
                                 | generate_queries_step_back 
                                 | retriever,
            # Pass on the question
            "question": lambda x: x['question']
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )
    
    with mlflow.start_run(run_name="RAG Chain Configuration - Step Back"):
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("retriever_type", "FAISS")
        mlflow.log_param("embedding_model", "nomic-embed-text:latest") 
    
    return chain


if __name__ == "__main__":
    st.title("RAG with query translation")
    
    # Upload de documentos
    uploads = st.file_uploader("Upload your documents here", accept_multiple_files=True)
    
    if uploads:
        retriever = config_retriever(uploads)
        
        # Configuração da LLM
        model_class = st.selectbox("Select the model class", ["groq", "hf_hub"])
        
        llm = config_llm(model_class)
        
        # Input do usuário
        with st.form(key="query_form"):
            user_query = st.text_input("Enter your question here")
            submit = st.form_submit_button("Send")

        if submit and user_query:
            # final_chain = multy_query_retrieval_chain(llm, retriever)
            # final_chain = multy_query_retrieval_chain_rag_fusion(llm, retriever)
            final_chain = step_back(llm, retriever, user_query)            
            with st.spinner("Generating answer..."):
                answer = final_chain.invoke({"question": user_query})
                # final_chain_and_answer = decomposition_chain(llm, retriever, user_query)
            
            st.subheader("Answer:")
            st.write(answer)