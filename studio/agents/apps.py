from django.apps import AppConfig


class AgentsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "agents"

    def ready(self) -> None:
        from . import signals  # noqa: F401  (registers signal handlers)
