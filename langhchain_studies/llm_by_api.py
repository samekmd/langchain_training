from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model_id = "llama-3.1-8b-instant"

llm = ChatGroq(
    api_key=api_key,
    model=model_id,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2 
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente e está respondendo perguntas gerais."),
    ("user", "Explique-me em 1 parágrafo o conceito de {topic}")
])

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    resp = chain.invoke({"topic": "astronomia"})
    print(resp)

    