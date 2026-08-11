// Studio (Django) owns agents + documents; Engine (FastAPI) owns chat.
// Paths go through the Vite dev proxy (see vite.config.js).

async function asJson(resPromise) {
  const res = await resPromise;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error || body.detail || `HTTP ${res.status}`);
  }
  return body;
}

export const listAgents = () => asJson(fetch("/studio/api/agents/"));

export const createAgent = (name, systemPrompt) =>
  asJson(
    fetch("/studio/api/agents/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, system_prompt: systemPrompt }),
    })
  );

export const uploadDocument = (slug, title, text) =>
  asJson(
    fetch(`/studio/api/agents/${slug}/documents/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, text }),
    })
  );

export const chat = (slug, question) =>
  asJson(
    fetch(`/engine/agents/${slug}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 4 }),
    })
  );
