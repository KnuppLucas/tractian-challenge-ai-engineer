import uuid
from django.db import models


class Document(models.Model):
    """
    Modelo responsável por armazenar informações sobre documentos enviados para processamento.
    - Cada registro representa um arquivo PDF processado pelo sistema.
    - Serve como entidade principal para relacionar os chunks de texto extraídos.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "documents"

    def __str__(self):
        return self.filename