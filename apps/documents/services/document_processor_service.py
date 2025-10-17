import os
import logging
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from apps.common.models.chunk_model import Chunk
from apps.common.models.document_model import Document

logger = logging.getLogger("rag_services")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class DocumentProcessorService:
    """
    Classe responsável por processar documentos PDF inseridos no sistema.
    - Extrai o texto de cada arquivo PDF
    - Divide o conteúdo em chunks textuais
    - Persiste os chunks e metadados no banco de dados
    """

    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process(self, files) -> dict:
        """
        Processa uma lista de arquivos PDF, extraindo o texto e criando chunks para indexação.

        :param files (list[UploadedFile]): Lista de arquivos PDF enviados pelo usuário.
        :return (dict): Dicionário contendo o total de documentos e chunks processados.
            - "documents_indexed" (int): Quantidade de documentos processados e salvos.
            - "total_chunks" (int): Total de chunks criados a partir de todos os documentos.
        """
        total_chunks, documents_indexed = 0, 0

        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            text = self.__extract_text(tmp_path)
            if not text.strip():
                logger.warning(f"⚠️ PDF {f.name} não possui texto extraído, possível PDF digitalizado.")
            chunks = self.__chunk_text(text)

            document = Document.objects.create(filename=f.name)
            for idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():
                    Chunk.objects.create(document=document, text=chunk_text, order=idx)
                    logger.debug(f"  • Chunk {idx} criado para Documento {f.name}")

            documents_indexed += 1
            total_chunks += len(chunks)
            os.remove(tmp_path)
            logger.info(f"✅ Documento {f.name} processado: {len(chunks)} chunks criados.")

        logger.info(f"📄 Total processado: {documents_indexed} documentos, {total_chunks} chunks.")
        return {"documents_indexed": documents_indexed, "total_chunks": total_chunks}

    def __extract_text(self, pdf_path):
        """
        Extrai o texto contido nas páginas de um arquivo PDF.

        :param pdf_path (str): Caminho temporário do arquivo PDF.
        :return (str): Texto concatenado extraído de todas as páginas do PDF.
        """
        text = ""
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            logger.debug(f"  • Página {page_num} extraída ({len(page_text)} caracteres)")
        return text

    def __chunk_text(self, text):
        """
        Divide o texto extraído do documento em múltiplos chunks menores para indexação.

        :param text (str): Texto completo extraído do PDF.
        :return (list[str]): Lista de chunks textuais gerados.
        """
        chunks = self.splitter.split_text(text)
        logger.debug(f"⚙️ Texto dividido em {len(chunks)} chunks.")
        return chunks