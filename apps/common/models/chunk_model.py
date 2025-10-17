import uuid
from django.db import models
from apps.common.models.document_model import Document


class Chunk(models.Model):
    """
    Modelo responsável por armazenar segmentos (chunks) de texto extraídos de documentos.
    - Cada chunk representa uma parte do texto de um documento processado.
    - Utilizado para operações de embedding e busca semântica.
    - Mantém a referência ao documento original e a ordem dos trechos.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, related_name="chunks", on_delete=models.CASCADE)
    text = models.TextField()
    order = models.IntegerField()

    class Meta:
        db_table = "chunks"

    def __str__(self):
        return f"Chunk {self.order} from {self.document.filename}"