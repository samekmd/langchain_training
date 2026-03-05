from langsmith import Client

from dotenv import load_dotenv
import os

load_dotenv()

client = Client(api_key=os.getenv("LANG_SMITH_API_KEY"))

prompt = client.pull_prompt("hwchase17/react")

if __name__ == "__main__":
    print(prompt)

