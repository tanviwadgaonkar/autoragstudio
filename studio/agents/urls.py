from django.urls import path

from . import views

urlpatterns = [
    path("agents/", views.agents_view, name="agents"),
    path("agents/<slug:slug>/documents/", views.documents_view, name="documents"),
    path(
        "agents/<slug:slug>/documents/<int:doc_id>/reindex/",
        views.reindex_view,
        name="reindex",
    ),
]
