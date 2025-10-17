
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from apps.documents.api.serializers import DocumentUploadSerializer
from apps.documents.services.embedding_service import EmbeddingService
from apps.documents.services.document_processor_service import DocumentProcessorService


class DocumentUploadView(APIView):
    """
    View responsável por disponibilizar o endpoint de upload de documentos.
    - Recebe arquivos PDF via requisição POST
    - Processa os PDFs em chunks de texto
    - Gera embeddings para os chunks e persiste no banco
    """

    def post(self, request):
        """
        Controla a rota POST para o upload de documentos, integrando processamento e embedding.

        :param request (Request): Objeto de requisição contendo os arquivos no campo 'files'.
        :return (Response): Retorna JSON com informações sobre o processamento ou erro.
            - status 202: Documentos processados e embeddings gerados com sucesso.
            - status 400: Nenhum arquivo foi enviado.
            - status 500: Erro interno ao processar documentos ou gerar embeddings.
        """
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        files = serializer.validated_data["files"]
        if not files:
            return Response(
                {"error": "Nenhum arquivo foi enviado. Use o campo 'files'."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            processor = DocumentProcessorService()
            result = processor.process(files)

            embedder = EmbeddingService()
            embedder.generate_and_store()

            return Response(
                {"message": "Documents processed successfully", **result},
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            print(e)
            return Response(
                {"error": f"Erro ao processar documentos: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )