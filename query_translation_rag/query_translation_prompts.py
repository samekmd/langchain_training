from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, load, loads

class QueryTranslationPromptTemplate():
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    
    def get_unique_union(self, documents: list[list]):
        """Unique union of retrieved docs"""
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Convert back to Document objects
        return [load(doc) for doc in unique_docs]
    

    def reciprocal_rank_fusion(self, documents: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}
        
        # Iterate through each list of ranked documents
        for docs in documents:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(docs)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (k + rank)
                fused_scores[doc_str] += 1 / (k + rank)
                
        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True) 
        ] 
        
        return reranked_results 
            
    def format_qa_pair(self, question, answer):
        """Format a question-answer pair for few-shot prompting."""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\n Answer: {answer}\n\n"
        return formatted_string.strip()
            
        
    def multi_query(self):
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide ONLY the five alternative questions separated by newlines, with no numbering, 
        no introductory text, and no blank lines. Original question: {question}
        """
        
        prompt_perspectives = ChatPromptTemplate.from_template(template)
        
        generate_queries = (
            prompt_perspectives
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") 
                  if q.strip()]) 
        )
        
        retrieval_chain = generate_queries | self.retriever.map() | self.get_unique_union
        
        return retrieval_chain
        
        
    def rag_fusion(self):
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
                    Generate multiple search queries related to: {question} \n
                    Output (4 queries):"""
        
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        
        generate_queries = (
            prompt_rag_fusion
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") 
                  if q.strip()]) 
        )
        
        retrieval_chain_rag_fusion = generate_queries | self.retriever.map() | self.reciprocal_rank_fusion
        
        return retrieval_chain_rag_fusion
    
    
    def decomposition(self, question):
        template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""
        
        prompt_decomposition = ChatPromptTemplate.from_template(template)
        
        generate_sub_questions = (
            prompt_decomposition
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") 
                  if q.strip()]) 
        ) 
        
        questions = generate_sub_questions.invoke({"question": question})
        
        return questions
    
    
    def step_back(self):
        examples = [
        {
            "input": "Qual foi a margem EBITDA da empresa X no último trimestre?",
            "output": "Como analisar a saúde financeira de uma empresa?",
        },
        {
            "input": "Por que as vendas caíram em março?",
            "output": "Quais fatores influenciam o desempenho de vendas de uma empresa?",
        },
        {
            "input": "O contrato com o fornecedor Y foi renovado?",
            "output": "Como funciona o processo de gestão de contratos com fornecedores?",
        },
]
        # We now transform these to example messages
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        
        prompt = ChatPromptTemplate.from_messages(
             [
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )
        
        generate_queries_step_back = prompt | self.llm | StrOutputParser()
        
        return generate_queries_step_back
        