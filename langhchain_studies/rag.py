from model import Model
from langchain_core.prompts import PromptTemplate
from datetime import date

def main():
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

    dia = date.today()

    # contexto = "Você sabe que hoje é dia '{}'".format(dia)
    
    contexto = """Faturamento trimestral:
                1º: R$42476,40
                2º: R$46212,97
                3º: R$41324,56
                4º: R$56430,24"""

    # pergunta = "Que dia é hoje? Retorne a data no formato dd/mm/yyyy"
    pergunta = "Que trimestre teve o maior faturamento?"

    model = class_model.chain
    
    res = model.invoke({"pergunta": pergunta, "contexto": contexto})

    print(res)
    

if __name__ == "__main__":
    main()