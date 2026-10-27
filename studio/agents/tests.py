"""Studio API tests. The engine is never contacted: ``requests`` is mocked."""

from __future__ import annotations

import json
from unittest import mock

import requests
from django.test import TestCase

from .models import Agent, Document


def _ok(payload: dict) -> mock.Mock:
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


class AgentApiTests(TestCase):
    def post(self, url: str, body) -> "HttpResponse":  # noqa: F821
        data = body if isinstance(body, str) else json.dumps(body)
        return self.client.post(url, data=data, content_type="application/json")

    def test_create_and_list_agent(self):
        resp = self.post("/api/agents/", {"name": "Support Bot", "system_prompt": "Be brief."})
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["slug"], "support-bot")
        agents = self.client.get("/api/agents/").json()["agents"]
        self.assertEqual([(a["slug"], a["documents"]) for a in agents], [("support-bot", 0)])

    def test_malformed_json_is_400_not_500(self):
        self.assertEqual(self.post("/api/agents/", "{not json").status_code, 400)
        self.assertEqual(self.post("/api/agents/", "[1, 2]").status_code, 400)

    def test_duplicate_name_is_409(self):
        self.post("/api/agents/", {"name": "Support Bot"})
        self.assertEqual(self.post("/api/agents/", {"name": "support bot"}).status_code, 409)

    def test_names_with_colliding_slugs_get_distinct_slugs(self):
        a = self.post("/api/agents/", {"name": "Foo Bar"}).json()["slug"]
        b = self.post("/api/agents/", {"name": "foo-bar"}).json()["slug"]
        self.assertEqual(a, "foo-bar")
        self.assertNotEqual(a, b)

    def test_non_latin_name_gets_a_usable_slug(self):
        first = self.post("/api/agents/", {"name": "कृषि सहायक"})
        second = self.post("/api/agents/", {"name": "मौसम सहायक"})
        self.assertEqual((first.status_code, second.status_code), (201, 201))
        slugs = {first.json()["slug"], second.json()["slug"]}
        self.assertEqual(len(slugs), 2)
        self.assertTrue(all(s.startswith("agent-") for s in slugs))


class DocumentIndexingTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(name="Docs Bot", system_prompt="Cite sources.")

    @mock.patch("agents.services.requests.post")
    def test_upload_indexes_document_and_sends_prompt(self, post):
        post.return_value = _ok({"chunks_indexed": 1})
        resp = self.client.post(
            "/api/agents/docs-bot/documents/",
            data=json.dumps({"title": "FAQ", "text": "Returns within 30 days."}),
            content_type="application/json",
        )
        self.assertEqual(resp.json()["status"], "indexed")
        url, kwargs = post.call_args.args[0], post.call_args.kwargs
        self.assertTrue(url.endswith("/agents/docs-bot/ingest"))
        self.assertEqual(kwargs["json"]["system_prompt"], "Cite sources.")

    @mock.patch("agents.services.requests.post", side_effect=requests.ConnectionError("refused"))
    def test_engine_down_marks_document_failed(self, _post):
        resp = self.client.post(
            "/api/agents/docs-bot/documents/",
            data=json.dumps({"title": "FAQ", "text": "Returns within 30 days."}),
            content_type="application/json",
        )
        body = resp.json()
        self.assertEqual(body["status"], "failed")
        self.assertIn("refused", body["last_error"])
        self.assertEqual(Document.objects.get().status, Document.Status.FAILED)

    @mock.patch("agents.services.requests.delete")
    def test_deleting_agent_drops_engine_index(self, delete):
        delete.return_value = _ok({"deleted": True})
        resp = self.client.delete("/api/agents/docs-bot/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Agent.objects.exists())
        self.assertTrue(delete.call_args.args[0].endswith("/agents/docs-bot/index"))

    @mock.patch("agents.services.requests.delete", side_effect=requests.ConnectionError("down"))
    def test_agent_delete_succeeds_even_if_engine_is_down(self, _delete):
        with self.assertLogs("agents.signals", level="WARNING"):
            self.assertEqual(self.client.delete("/api/agents/docs-bot/").status_code, 200)
        self.assertFalse(Agent.objects.exists())


class CrossSiteRequestTests(TestCase):
    """CSRF-exempt endpoints must not accept the "simple" requests a foreign page can send without a preflight."""

    def test_text_plain_post_is_rejected(self):
        resp = self.client.post("/api/agents/", data='{"name": "Evil"}', content_type="text/plain")
        self.assertEqual(resp.status_code, 415)
        self.assertFalse(Agent.objects.exists())

    def test_form_post_is_rejected(self):
        resp = self.client.post("/api/agents/", data={"name": "Evil"})  # multipart form
        self.assertEqual(resp.status_code, 415)
        self.assertFalse(Agent.objects.exists())
