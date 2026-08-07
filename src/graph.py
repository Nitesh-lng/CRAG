from langgraph.graph import StateGraph, START, END
from src.state import GraphState
from src.nodes import transform_query, retrieve, grade_documents, web_search, generate, decide_to_generate

workflow = StateGraph(GraphState)

workflow.add_node("transform_query", transform_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",       
    decide_to_generate,      
    {
        "web_search": "web_search",
        "generate": "generate",    
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()