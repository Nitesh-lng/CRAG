from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_results: List[Document]
    generation: str
    web_search: bool
    source: str
