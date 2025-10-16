from django.urls import path

from apps.documents.api.views import DocumentUploadView

urlpatterns = [
    path("documents/", DocumentUploadView.as_view(), name="upload-documents"),
]
