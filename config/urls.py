from django.urls import path, include

urlpatterns = [
    path("api/", include("apps.documents.api.urls")),
    path("api/", include("apps.questions.api.urls")),
]