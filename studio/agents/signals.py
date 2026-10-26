"""Keep the engine's vector index in step with the Studio's agent registry."""

from __future__ import annotations

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Agent
from .services import EngineError, drop_agent_index

log = logging.getLogger(__name__)


@receiver(post_delete, sender=Agent)
def drop_index_for_deleted_agent(sender, instance: Agent, **kwargs) -> None:
    # Best effort: the Studio row is already gone, and a down engine must not
    # block deletes. The failure is logged so the orphan can be cleaned up.
    try:
        drop_agent_index(instance.slug)
    except EngineError as exc:
        log.warning("could not drop engine index for agent %s: %s", instance.slug, exc)
