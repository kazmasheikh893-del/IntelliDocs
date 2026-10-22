from django.urls import path
from . import views

app_name = "documents"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("upload/", views.upload_document_view, name="upload"),
    path("qa/", views.qa_view, name="qa"),
    path("api/query/", views.query_api_view, name="api_query"),
    path("document/<int:document_id>/delete/", views.delete_document_view, name="delete"),
]

