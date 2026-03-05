from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from .get_transcription import get_transcription
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def model_groq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=2048):
    llm = ChatGroq(
        api_key=API_KEY,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm


def create_prompt():
    system_message = "Você é um assistente virtual prestativo e deve responder a uma consulta com base na transcrição de um vídeo."

    user_message = "Consulta: {consulta} \n Transcrição: {transcricao}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_message)
    ])

    return prompt_template

def llm_chain():
    prompt_template = create_prompt()
    llm = model_groq()
    
    chain = prompt_template | llm | StrOutputParser()
    
    return chain

def interpret_video(consulta, url):
    chain = llm_chain()
    transcricao = get_transcription(url)
    
    res = chain.invoke({"transcricao": transcricao, "consulta": consulta})
    
    return res

if __name__ == "__main__":
    chain = llm_chain()
    
    res = interpret_video("Qual é o tema do vídeo?", "https://www.youtube.com/watch?v=tXM3Ifd6_T8")
    
    print(res)