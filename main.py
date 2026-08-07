import logging

from src.vector_store import VectorStoreBuilder
from src.loader import TextDocumentLoader
from src.chunker import TextDocumentSplitter
from src.config import PDF_PATH
from src.graph import app
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def save_graph_png(path: str = "graph.png"):
    """Render the compiled LangGraph as a Mermaid PNG for documentation."""
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        logger.info("Graph diagram saved to %s", path)
    except Exception as exc:
        logger.warning(
            "Could not render graph diagram (%s). "
            "Install the drawing extras to enable it, e.g. "
            "`pip install 'langgraph[draw]'` or ensure graphviz/mermaid deps are available.",
            exc,
        )


def main():
    builder = VectorStoreBuilder()
    index_path = "faiss_index"

    if Path(index_path).exists():
        logger.info("Loading existing vector store...")
        builder.load(index_path)
        logger.info("Vector store ready!")
    else:
        logger.info("Building vector store...")
        loader = TextDocumentLoader(PDF_PATH)
        documents = loader.load()

        splitter = TextDocumentSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.chunk(documents)

        vector_store = builder.build(chunks)
        builder.save(vector_store, index_path)
        logger.info("Vector store saved.")

    # Generate the graph visualization (best-effort).
    save_graph_png("graph.png")

    logger.info("-" * 50)

    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query == "exit":
            logger.info("Leaving")
            break

        result = app.invoke({"question": query})

        print(f"\nQuestion: {query}")
        print(f"Answer: {result['generation']}")
        print(f"Source: {result.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
