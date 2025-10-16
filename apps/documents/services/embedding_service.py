import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from apps.documents.models.chunk_model import Chunk
from apps.documents.models.embedding_model import Embedding


class EmbeddingService:
    """
    Classe responsável por gerar e armazenar embeddings de chunks de texto.
    - Extrai os embeddings de cada chunk
    - Persiste os vetores no banco de dados
    - Cria e salva o índice FAISS para busca vetorial
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss_index.index"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index_path = index_path
        self.index = None

    def generate_and_store(self):
        """
        Gera os embeddings dos chunks armazenados no banco e salva tanto no banco quanto no índice FAISS.

        :return (None): Operação de geração e salvamento executada com sucesso.
        """
        chunks = Chunk.objects.all()
        vectors, ids = [], []

        for chunk in chunks:
            vector = self.embeddings.embed_query(chunk.text)
            vector_np = np.array(vector, dtype="float32")
            vectors.append(vector_np)
            ids.append(chunk.id.bytes)
            Embedding.objects.create(chunk=chunk, vector=vector_np.tobytes())

        self.__save_faiss_index(vectors, ids)

    def __save_faiss_index(self, vectors, ids):
        """
        Salva os vetores gerados em um índice FAISS no disco.

        :param vectors (list[np.ndarray]): Lista de vetores de embeddings a serem indexados.
        :param ids (list[bytes]): Lista de identificadores binários correspondentes aos vetores.
        :return (None): Cria e salva o índice FAISS no caminho especificado.
        """
        if not vectors:
            print("⚠️ Nenhum vetor para indexar — verifique se há chunks.")
            return

        vectors = np.vstack(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, self.index_path)
        self.index = index