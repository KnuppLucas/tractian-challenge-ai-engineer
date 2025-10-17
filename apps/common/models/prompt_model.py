import uuid
from django.db import models
from django.utils import timezone


class Prompt(models.Model):
    """
    Model para armazenar prompts e suas respostas no pipeline RAG.
    Permite auditoria, análise de performance e histórico de consultas.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.TextField(help_text="Pergunta feita ao sistema")
    context = models.TextField(help_text="Contexto usado na geração (chunks combinados)", blank=True, null=True)
    generated_prompt = models.TextField(help_text="Prompt final enviado ao LLM", blank=True, null=True)
    answer = models.TextField(help_text="Resposta gerada pelo LLM", blank=True, null=True)
    references_list = models.JSONField(help_text="Referências dos chunks usados", default=list, blank=True)
    model_name = models.CharField(max_length=128, help_text="Modelo LLM usado", default="unknown")
    embedding_model = models.CharField(max_length=128, help_text="Modelo de embeddings usado", default="unknown")
    created_at = models.DateTimeField(default=timezone.now)
    latency_ms = models.FloatField(help_text="Tempo total de processamento em milissegundos", null=True, blank=True)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "prompt"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Prompt({self.id}) - {self.question[:40]}..."