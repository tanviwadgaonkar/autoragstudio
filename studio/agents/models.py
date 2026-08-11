"""Agent registry and document store.

The studio is the source of truth for *what agents exist* and *what documents
they own*. The engine is the source of truth for the vector index. The two
are linked only by the agent slug, which the engine uses as its collection
key — so deleting an agent here and calling the engine's DELETE endpoint
fully removes it everywhere.
"""

from django.db import models
from django.utils.text import slugify


class Agent(models.Model):
    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    description = models.CharField(max_length=300, blank=True)
    system_prompt = models.TextField(
        blank=True,
        help_text="Sent to the engine on ingest; used as the agent's persona.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.name


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        INDEXED = "indexed", "Indexed"
        FAILED = "failed", "Failed"

    agent = models.ForeignKey(
        Agent, on_delete=models.CASCADE, related_name="documents"
    )
    title = models.CharField(max_length=200)
    text = models.TextField()
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.PENDING
    )
    last_error = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["agent", "title"], name="unique_doc_title_per_agent"
            )
        ]

    def __str__(self) -> str:
        return f"{self.title} ({self.agent.slug})"
