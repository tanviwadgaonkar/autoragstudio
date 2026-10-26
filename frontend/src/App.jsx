import { useEffect, useState } from "react";
import { chat, createAgent, deleteAgent, listAgents, uploadDocument } from "./api";

export default function App() {
  const [agents, setAgents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = () =>
    listAgents()
      .then((data) => setAgents(data.agents))
      .catch((e) => setError(e.message));

  useEffect(() => {
    refresh();
  }, []);

  const onCreateAgent = async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    try {
      const agent = await createAgent(
        form.get("name"),
        form.get("system_prompt")
      );
      e.target.reset();
      await refresh();
      setSelected(agent.slug);
    } catch (err) {
      setError(err.message);
    }
  };

  const onUpload = async (e) => {
    e.preventDefault();
    if (!selected) return;
    const form = new FormData(e.target);
    try {
      const doc = await uploadDocument(
        selected,
        form.get("title"),
        form.get("text")
      );
      e.target.reset();
      setError(
        doc.status === "indexed"
          ? ""
          : `Document saved but not indexed (${doc.last_error || doc.status}) — is the engine running?`
      );
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  const onDeleteAgent = async () => {
    if (!selected || !window.confirm(`Delete agent "${selected}" and its index?`)) return;
    try {
      await deleteAgent(selected);
      setSelected(null);
      setMessages([]);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  const onAsk = async (e) => {
    e.preventDefault();
    if (!selected || !question.trim()) return;
    const q = question.trim();
    setQuestion("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: q }]);
    try {
      const res = await chat(selected, q);
      setMessages((m) => [
        ...m,
        { role: "agent", text: res.answer, sources: res.sources },
      ]);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="layout">
      <aside>
        <h1>AutoRAG Studio</h1>
        <p className="tag">Multi-agent RAG · local LLMs</p>

        <h2>Agents</h2>
        <ul className="agents">
          {agents.map((a) => (
            <li key={a.slug}>
              <button
                className={a.slug === selected ? "active" : ""}
                onClick={() => {
                  setSelected(a.slug);
                  setMessages([]);
                }}
              >
                {a.name} <span className="count">{a.documents} docs</span>
              </button>
            </li>
          ))}
        </ul>

        <form onSubmit={onCreateAgent} className="stack">
          <h2>New agent</h2>
          <input name="name" placeholder="Agent name" required />
          <textarea
            name="system_prompt"
            placeholder="System prompt (optional)"
            rows={3}
          />
          <button type="submit">Create</button>
        </form>

        {selected && (
          <form onSubmit={onUpload} className="stack">
            <h2>Add document → {selected}</h2>
            <input name="title" placeholder="Document title" required />
            <textarea name="text" placeholder="Paste text…" rows={5} required />
            <button type="submit">Ingest</button>
            <button type="button" className="danger" onClick={onDeleteAgent}>
              Delete agent
            </button>
          </form>
        )}
      </aside>

      <main>
        {error && <div className="error">{error}</div>}
        {!selected && <p className="hint">Select or create an agent to chat.</p>}
        <div className="chat">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <p>{m.text}</p>
              {m.sources?.length > 0 && (
                <details>
                  <summary>{m.sources.length} source(s)</summary>
                  {m.sources.map((s, j) => (
                    <blockquote key={j}>
                      <b>
                        {s.metadata.doc_title} · score {s.score.toFixed(3)}
                      </b>
                      <p>{s.text}</p>
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          ))}
          {busy && <div className="msg agent">…</div>}
        </div>
        {selected && (
          <form onSubmit={onAsk} className="ask">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={`Ask ${selected}…`}
            />
            <button disabled={busy} type="submit">
              Send
            </button>
          </form>
        )}
      </main>
    </div>
  );
}
