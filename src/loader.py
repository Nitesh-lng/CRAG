from langchain_community.document_loaders import PyPDFLoader

class TextDocumentLoader:

    def __init__(self,file_path):
        self.file_path=file_path

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError('File not found!')
        if self.file_path.suffix.lower() != '.pdf':
            raise ValueError('Extension issue.. ')
        
        loader=PyPDFLoader(self.file_path)
        documents=loader.load()

        if not documents:
            raise ValueError('File is Empty..! ')

        return documents