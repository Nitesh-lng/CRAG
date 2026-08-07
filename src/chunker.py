from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextDocumentSplitter:

    def __init__(self,chunk_size=1000,chunk_overlap=200):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size
        )

    def chunk(self,document):
        chunks=self.splitter.split_documents(document)
        return chunks
        