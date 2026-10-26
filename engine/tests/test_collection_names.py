"""Agent id -> collection name mapping: the isolation boundary."""

import re

import pytest

from app.rag.store import collection_name

CHROMA_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]")


def test_short_slugs_keep_their_v1_name():
    # Unchanged naming means indexes built before this change still load.
    assert collection_name("support-bot") == "agent--support-bot"


@pytest.mark.parametrize(
    "agent_id",
    [
        "x" * 55 + "-yz",  # v1 truncated this to a trailing '-' (invalid in Chroma)
        "customer-support-knowledge-base-for-enterprise-clients-region-a",
        "Support Bot!",
        "a" * 200,
    ],
)
def test_names_are_always_valid_chroma_names(agent_id):
    assert CHROMA_NAME.fullmatch(collection_name(agent_id))


def test_distinct_ids_never_collide():
    ids = [
        "customer-support-knowledge-base-for-enterprise-clients-region-a",
        "customer-support-knowledge-base-for-enterprise-clients-region-b",
        "Support-Bot",
        "support-bot",
        "support bot",
        "x" * 100,
        "x" * 101,
    ]
    names = [collection_name(i) for i in ids]
    assert len(set(names)) == len(ids)


def test_id_without_alphanumerics_is_rejected():
    with pytest.raises(ValueError):
        collection_name("!!!")
