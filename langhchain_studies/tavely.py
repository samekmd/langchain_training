from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
import os

load_dotenv()

TAVELY_API_KEY = os.getenv("TAVELY_API_KEY")

search = TavilySearchResults(max_results=2)

search_results = search.invoke("Quando foi a última Copa do Mundo?")

if __name__ == "__main__":
    print(search_results)