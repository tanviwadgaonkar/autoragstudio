from django.contrib import admin

from .models import Agent, Document


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "description", "created_at")
    prepopulated_fields = {"slug": ("name",)}
    search_fields = ("name", "slug")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "agent", "status", "created_at")
    list_filter = ("status", "agent")
    search_fields = ("title",)
