from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

class Model:
    def __init__(self):
        self.model = "phi3:mini"

        self.llm = ChatOllama(
            model=self.model,
            temperature=0.1,
            repeat_penalty=1.1
        )

        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente e está respondendo perguntas gerais."),
            ("user", "Explique-me em 1 parágrafo o conceito de {topic}")
        ])

        self.count = RunnableLambda(lambda x: f"Palavras: {len(x.split())}\n{x}")
        
        self._chain = self._prompt | self.llm | StrOutputParser() 
        
    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        self._chain = self._prompt | self.llm | StrOutputParser()
        
    @property
    def chain(self):
        return self._chain
    
    @chain.setter
    def chain(self, value):
        self._chain = value
    
    def run(self, topic):
        chain = self.prompt | self.llm | StrOutputParser() | self.count
        resp = chain.invoke({"topic": topic})
        return resp
    
    def stream(self, topic):
        chain = self.prompt | self.llm | StrOutputParser() | self.count
        for chunk in chain.stream({"topic": topic}):
            yield chunk
            
        