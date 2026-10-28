from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentVectorStore:
    document_path: Path
    db_path: Path
    collection_name: str

    embeddings: OllamaEmbeddings
    vectorstore: Chroma

    def __init__(
            self,
            document_path: Path,
            db_path: Path,
            collection_name: str):

        self.document_path = document_path
        self.db_path = db_path
        self.collection_name = collection_name

        self.load()

    def rebuild_database(self) -> Chroma:
        # Load the document
        loader = TextLoader(
            str(self.document_path),
            encoding="utf-8"
            )

        documents = loader.load()

        if not documents:
            raise ValueError("No text was extracted from the document")

        # Split into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )

        chunks = splitter.split_documents(documents)

        # Create and persist the vector database
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.db_path),
        )

        return vectorstore

    def get_database(self) -> Chroma:
        if not self.db_path.exists():
            # Create the database as it does not yet exist
            return self.rebuild_database()
        # Load the pre-existing database
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_path)
        )

    def load(self):
        # Create the embeddings model
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Get or create the vectorstore
        self.vectorstore = self.get_database()

    def similarity_search(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search(query, k)

if __name__ == '__main__':
    dvs = DocumentVectorStore(
        Path('./documents/CRG_Documentation.md'),
        Path('./chroma_db'),
        'crystal_rock_docs')

    print(dvs.vectorstore.similarity_search("Operating temperature DW-4", k=4))