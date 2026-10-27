"""Agent registry and document store.

The studio is the source of truth for *what agents exist* and *what documents
they own*. The engine is the source of truth for the vector index. The two
are linked only by the agent slug, which the engine uses as its collection
key. Deleting an agent here also drops its engine index (see signals.py),
so a new agent with the same name never inherits the old one's documents.
"""

import uuid

from django.db import models
from django.utils.text import slugify


def unique_slug_for(name: str, *, exclude_pk: int | None = None) -> str:
    """A slug that is non-empty and unused.

    ``slugify`` drops non-Latin text entirely ("कृषि सहायक" -> ""), and distinct
    names can share a slug ("Foo Bar" / "foo-bar"), so fall back to a random
    stem and de-duplicate with a numeric suffix.
    """
    base = slugify(name)[:120] or f"agent-{uuid.uuid4().hex[:8]}"
    candidate, n = base, 2
    others = Agent.objects.exclude(pk=exclude_pk) if exclude_pk else Agent.objects.all()
    while others.filter(slug=candidate).exists():
        candidate, n = f"{base}-{n}", n + 1
    return candidate


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
            self.slug = unique_slug_for(self.name, exclude_pk=self.pk)
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
