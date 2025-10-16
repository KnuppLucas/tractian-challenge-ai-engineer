from django.urls import path
from apps.questions.api.views import QuestionView

urlpatterns = [
    path("question/", QuestionView.as_view(), name="ask-question"),
]