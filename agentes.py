from langchain.agents import create_agent
from langchain_groq import ChatGroq
from tools import tools
from ReAct import prompt

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model_id = "llama-3.3-70b-versatile"

model = ChatGroq(
    api_key=api_key,
    model=model_id,
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=True
)


agent = create_agent(model=model, 
                     tools=tools, 
                    system_prompt="You are a helpful assistant. Be concise and accurate."
                     )


def main(question: str):
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    return result['messages'][-1].content

if __name__ == "__main__":
    question = "How many Champions League titles Real Madrid has?"
    res = main(question)
    print(res)