from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = "phi3:mini"

llm = ChatOllama(
    model=model,
    temperature=0.1,
    repeat_penalty=1.1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente e está respondendo perguntas gerais."),
    ("user", "Explique-me em 1 parágrafo o conceito de {topic}")
])

# chain =  prompt | llm 

# chain_str = chain | StrOutputParser()

# resp = chain_str.invoke({"topic": "astronomia"})

count = RunnableLambda(lambda x: f"Palavras: {len(x.split())}\n{x}")

chain = prompt | llm | StrOutputParser() | count

resp = chain.invoke({"topic": "astronomia"})

if __name__ == "__main__":
    print(resp)