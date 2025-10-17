from rest_framework import serializers

class QuestionSerializer(serializers.Serializer):
    """
    Classe responsável por fazer a Serialização da Questão recebida na nossa API.
    """
    question = serializers.CharField()