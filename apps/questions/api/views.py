from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from apps.questions.api.serializers import QuestionSerializer
from apps.questions.services.rag_service import RAGService

class QuestionView(APIView):
    """
    View responsável por expor a rota de perguntas integrando com o serviço RAG.
    - Recebe uma questão via requisição POST
    - Processa o texto usando o RAGService
    - Retorna a resposta e suas referências
    """

    def post(self, request):
        """
        Controla a rota POST para o endpoint de question, executando o fluxo completo do RAG.

        :param request (Request): Objeto de requisição contendo o campo 'question' no corpo (JSON).
        :return (Response): Retorna JSON com a resposta gerada e referências, ou erro interno.
            - status 200: Sucesso na geração da resposta
            - status 500: Erro interno no processamento
        """
        serializer = QuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]
        try:
            rag = RAGService()
            result = rag.ask(question)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
