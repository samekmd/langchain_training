from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=3000
))

@tool
def wikipedia_tool(query: str) -> str:
    """Use this tool to search for information on Wikipedia. Input should be a search query."""
    return wikipedia.run(query)

@tool
def date_tool() -> str:
    """Use this tool to get the current day."""
    from datetime import date
    return date.today().strftime("%d/%m/%Y")

tools = [wikipedia_tool, date_tool]

if __name__ == "__main__":
    print(wikipedia.run("Deep Learning"))