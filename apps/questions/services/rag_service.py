import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from apps.documents.models.chunk_model import Chunk


login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


class RAGService:
    """
    Serviço RAG robusto com reranking baseado em embeddings.
    - Carrega ou cria FAISS index
    - Realiza reranking semântico pós-retrieval
    - Suporta fallback de modelo e log detalhado
    """

    def __init__(
        self,
        index_dir="faiss_index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="google/long-t5-tglobal-base",
        fallback_model="google/flan-t5-base",
    ):
        self.index_dir = index_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = self.__load_or_create_faiss_index()
        self.llm_model_name = llm_model
        self.fallback_model_name = fallback_model
        self.pipe = self.__build_pipeline()
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def __build_pipeline(self):
        """
        Constrói o pipeline de geração de texto com o modelo principal ou o fallback.

        :return (transformers.Pipeline): Pipeline configurado para geração de texto.
        """
        try:
            print(f"⚙️ Carregando modelo principal: {self.llm_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.llm_model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=False,
            )
        except Exception as e:
            print(f"⚠️ Falha ao carregar {self.llm_model_name}: {e}")
            print(f"➡️ Usando fallback: {self.fallback_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_model_name)

        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def __load_or_create_faiss_index(self):
        """
        Carrega o índice FAISS existente do disco ou cria um novo a partir dos chunks no banco.

        :raises ValueError: Caso não existam chunks disponíveis no banco para criar o índice.
        :return (FAISS): Objeto FAISS configurado e pronto para consultas.
        """
        if os.path.exists(self.index_dir):
            print(f"✅ Carregando índice existente de {self.index_dir}")
            return FAISS.load_local(
                folder_path=self.index_dir,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

        print("⚙️ Criando novo índice FAISS a partir dos chunks do banco...")
        chunks = Chunk.objects.all()
        if not chunks:
            raise ValueError("Nenhum chunk encontrado no banco para criar FAISS index.")

        docs = [
            Document(page_content=chunk.text, metadata={"chunk_id": str(chunk.id)})
            for chunk in chunks
        ]
        vectorstore = FAISS.from_documents(docs, embedding=self.embedding_model)
        vectorstore.save_local(self.index_dir)
        print("✅ Novo índice FAISS criado e salvo.")
        return vectorstore

    def __rerank_chunks(self, question: str, candidate_docs: list, top_k_final: int = 5):
        """
        Reranqueia chunks recuperados com base na similaridade de embeddings com a pergunta.

        :param question (str): Pergunta do usuário usada como base para o reranking.
        :param candidate_docs (list[Document]): Lista de documentos candidatos retornados pelo retriever.
        :param top_k_final (int): Quantidade de documentos finais a manter após o reranking.
        :return (list[Document]): Lista dos melhores documentos reranqueados.
        """
        if not candidate_docs:
            return []

        query_vec = np.array(
            self.embedding_model.embed_query(question), dtype=np.float32
        ).reshape(1, -1)

        chunk_vecs = np.vstack(
            [
                np.array(
                    self.embedding_model.embed_query(doc.page_content), dtype=np.float32
                )
                for doc in candidate_docs
            ]
        )

        sims = 1 - cdist(query_vec, chunk_vecs, metric="cosine")[0]
        top_indices = np.argsort(sims)[-top_k_final:][::-1]

        reranked = [(candidate_docs[i], sims[i]) for i in top_indices]

        print("\n🔍 Reranking de chunks (maiores similaridades primeiro):")
        for doc, score in reranked:
            cid = doc.metadata.get("chunk_id")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  • Chunk {cid} | Score: {score:.4f} | {preview}...")

        return [doc for doc, _ in reranked]

    def ask(
        self,
        question: str,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
        log_chunks: bool = True,
    ) -> dict:
        """
        Realiza o fluxo completo RAG: recuperação, reranking e geração de resposta textual.

        :param question (str): Pergunta do usuário.
        :param top_k_retrieve (int): Número de documentos recuperados inicialmente pelo retriever.
        :param top_k_final (int): Quantidade final de chunks após o reranking semântico.
        :param log_chunks (bool): Se True, imprime logs com os chunks utilizados na resposta.
        :return (dict): Dicionário com a resposta gerada e até 3 referências de contexto.
            - "answer" (str): Texto final gerado pelo modelo.
            - "references" (list[str]): Trechos dos chunks usados na geração.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
        candidate_docs = retriever.invoke(question)

        reranked_docs = self.__rerank_chunks(question, candidate_docs, top_k_final)

        context_text = "\n\n".join([doc.page_content for doc in reranked_docs])
        prompt = (
            f"Responda à pergunta abaixo com base no contexto fornecido.\n\n"
            f"Contexto:\n{context_text}\n\n"
            f"Pergunta: {question}\n\nResposta:"
        )

        try:
            result = self.llm.invoke(prompt)
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                answer = result[0]["generated_text"].strip()
            elif isinstance(result, str):
                answer = result.strip()
            else:
                answer = str(result)
        except Exception as e:
            print(f"❌ Erro ao gerar resposta com LLM: {e}")
            answer = "Erro interno ao processar a resposta do modelo."

        valid_docs = [
            doc
            for doc in reranked_docs
            if Chunk.objects.filter(id=doc.metadata.get("chunk_id")).exists()
        ]

        if log_chunks:
            print("\n📄 Chunks usados no prompt final:")
            for doc in valid_docs:
                print(f"- ID: {doc.metadata.get('chunk_id')}, Preview: {doc.page_content[:100]}...")

        refs = [doc.page_content.strip().replace("\n", " ")[:500] for doc in valid_docs]

        if not answer or not valid_docs:
            answer = "Desculpe, não encontrei uma resposta relevante com base nos documentos."

        return {"answer": answer, "references": refs[:3]}
