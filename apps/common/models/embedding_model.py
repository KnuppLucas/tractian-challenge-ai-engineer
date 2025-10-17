import uuid
from django.db import models
from apps.common.models.chunk_model import Chunk


class Embedding(models.Model):
    """
    Modelo responsável por armazenar os vetores de embeddings associados a cada chunk de texto.
    - Cada embedding corresponde a um único chunk
    - O vetor é armazenado em formato binário
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chunk = models.OneToOneField(Chunk, on_delete=models.CASCADE, related_name="embedding")
    vector = models.BinaryField()

    class Meta:
        db_table = "embeddings"