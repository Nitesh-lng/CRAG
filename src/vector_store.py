from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreBuilder:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            encode_kwargs={"normalize_embeddings": True}
        )

    def build(self,chunks):

        vector_store= FAISS.from_documents(
            chunks,
            self.embeddings
        )
        return vector_store

    def save(self, vector_store, path):          
        vector_store.save_local(path)

    def load(self,path):
        return FAISS.load_local(
            path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def get_retriever(self, vector_store, k=3, score_threshold=0.5):
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
            },
        )