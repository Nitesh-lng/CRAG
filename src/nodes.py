import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

builder = VectorStoreBuilder()
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = builder.load("faiss_index")
    return _vector_store

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Document relevant to question, 'yes' or 'no'")

llm=ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0
)

web_search_tool = TavilySearchResults(k=3)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document is relevant to a user question.

The document is relevant ONLY if it can actually help answer the specific question asked.
Do not grade as relevant just because the document shares a keyword or topic with the question.
Consider whether the document actually addresses what the question is asking for.

Give a binary score 'yes' or 'no'.

Retrieved document:
{document}

User question:
{question}"""
)

transform_prompt = ChatPromptTemplate.from_template(
    """You are a query rewriting assistant. Rewrite the user's question into a
cleaner, self-contained, search-optimized query that will retrieve the most
relevant documents.

Rules:
- Keep the original intent and meaning intact.
- Remove filler, ambiguity, and conversational phrasing.
- Do NOT answer the question.
- Return ONLY the rewritten query, with no preamble, quotes, or explanation.

Original question:
{question}

Rewritten query:"""
)

gen_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
    If the context does not contain the answer, say "I don't have enough information to answer that."

    Context:
    {context}

    Question: {question}

    Answer:"""
    )

rag_chain = gen_prompt | llm | StrOutputParser()
retrieval_grader = grade_prompt | structured_llm_grader
query_rewriter = transform_prompt | llm | StrOutputParser()

def transform_query(state):
    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    rewritten_query = query_rewriter.invoke({"question": question}).strip()
    logger.info("Rewritten query: %s", rewritten_query)
    return {"question": rewritten_query}

def retrieve(state):
    question = state["question"]
    documents = get_vector_store().similarity_search(question, k=3)
    logger.info("---RETRIEVE---")
    return {"documents": documents}

def grade_documents(state):
    logger.info("---GRADE---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        result = retrieval_grader.invoke(
            {"document": doc.page_content, "question": question}
        )
        grade = result.binary_score

        if grade.lower() == "yes":
            logger.info("Doc graded: RELEVANT")
            filtered_docs.append(doc)
        else:
            logger.info("Doc graded: NOT RELEVANT")
            web_search = True

    source = "local" if filtered_docs else "none"

    return {"documents": filtered_docs, "web_search": web_search, "source": source}

def decide_to_generate(state):
    if state["web_search"]:
        return "web_search"
    else:
        return "generate"

def web_search(state):
    query=state['question']
    result= web_search_tool.invoke(query)
    web_docs = [Document(page_content=res['content']) for res in result]
    logger.info("---WEB SEARCH---")

    # If local docs also survived grading, the answer draws on both sources.
    source = "local + web" if state.get("documents") else "web"

    return {'web_results':web_docs, "source": source}

def generate(state):

    query=state['question']
    docs=state['documents']
    web=state.get('web_results',[])

    all_docs=docs+web
    context= '\n\n'.join(doc.page_content for doc in all_docs)

    generation = rag_chain.invoke({"context": context, "question": query})
    logger.info("---GENERATE---")
    return {"generation": generation}