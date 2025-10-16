from rest_framework import serializers

class DocumentUploadSerializer(serializers.Serializer):
    """Valida o upload de PDFs."""
    files = serializers.ListField(
        child=serializers.FileField(), allow_empty=False, help_text="Lista de arquivos PDF."
    )