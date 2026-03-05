import streamlit as st 
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from youtube_transcript.main import model_groq

from dotenv import load_dotenv

load_dotenv()

HUGGING_FACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



#meta-llama/Meta-Llama-3-8b-Instruct
# mistralai/Mistral-7B-Instruct-v0.2
def model_hf_hub(model='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=512,
        task="text-generation",
        huggingfacehub_api_token=HUGGING_FACEHUB_API_TOKEN,
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    return chat_model

def model_response(user_query, chat_history, model_class, language="português"):
    llm_dict = {
        "groq": model_groq(),
        "hf_hub": model_hf_hub()
    }
    
    llm = llm_dict[model_class]
    
    # Definição dos prompts 
    system_prompt = """
        Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}. 
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_query": user_query,
        "language": language
    })
    

     
if __name__ == "__main__":
    # Configuração do streamlit, with emojis
    st.set_page_config(page_title="Seu assistente virtual", page_icon="🤖")
    st.title("Seu assistente virtual com memória 🤖")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage(content="Olá! Como posso ajudar você hoje?")]
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
                
    user_query = st.chat_input("Digite sua mensagem")       
    
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class="hf_hub"))
            
        st.session_state.chat_history.append(AIMessage(content=resp))