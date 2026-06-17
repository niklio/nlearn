/**
 * nlearn leaderboard — Cloudflare Worker API.
 *
 * Routes (everything else falls through to the static frontend in public/):
 *   GET  /api/boards            → list of valid board ids
 *   GET  /api/:board            → { board, entries: [...], updated_at }
 *   POST /api/:board            → upsert one entry (Bearer auth)
 *
 * Storage: one KV key per board ("board:<id>") holding a JSON array of entries.
 * Entries are upserted by `id`, so a training run or a kernel version updates
 * its own row in place rather than appending duplicates.
 */

const BOARDS = ["pretraining", "flashattention", "gemm"];
const MAX_ENTRIES = 200; // per board; oldest-updated rows evicted past this

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "content-type, authorization",
  "cache-control": "no-store",
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: JSON_HEADERS });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Only /api/* is handled here; static assets are served automatically.
    if (!path.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: JSON_HEADERS });
    }

    if (path === "/api/boards") {
      return json({ boards: BOARDS });
    }

    const board = path.slice("/api/".length).replace(/\/+$/, "");
    if (!BOARDS.includes(board)) {
      return json({ error: `unknown board '${board}'`, boards: BOARDS }, 404);
    }
    const key = `board:${board}`;

    if (request.method === "GET") {
      const raw = await env.LEADERBOARD_KV.get(key);
      const entries = raw ? JSON.parse(raw) : [];
      return json({ board, entries, updated_at: new Date().toISOString() });
    }

    if (request.method === "POST") {
      const auth = request.headers.get("authorization") || "";
      const token = auth.replace(/^Bearer\s+/i, "");
      if (!env.LEADERBOARD_TOKEN || token !== env.LEADERBOARD_TOKEN) {
        return json({ error: "unauthorized" }, 401);
      }

      let entry;
      try {
        entry = await request.json();
      } catch {
        return json({ error: "invalid JSON body" }, 400);
      }
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        return json({ error: "body must be a JSON object" }, 400);
      }
      if (typeof entry.id !== "string" || !entry.id) {
        return json({ error: "entry.id (string) is required" }, 400);
      }
      if (!entry.metrics || typeof entry.metrics !== "object") {
        return json({ error: "entry.metrics (object) is required" }, 400);
      }

      // Server-stamped fields override anything the client sent.
      entry.updated_at = new Date().toISOString();
      entry.name = String(entry.name || entry.id);
      entry.status = entry.status === "running" ? "running" : "done";

      const raw = await env.LEADERBOARD_KV.get(key);
      let entries = raw ? JSON.parse(raw) : [];

      const idx = entries.findIndex((e) => e.id === entry.id);
      if (idx >= 0) {
        // Preserve original creation time across updates.
        entry.created_at = entries[idx].created_at || entry.updated_at;
        entries[idx] = entry;
      } else {
        entry.created_at = entry.updated_at;
        entries.push(entry);
      }

      if (entries.length > MAX_ENTRIES) {
        entries.sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1));
        entries = entries.slice(0, MAX_ENTRIES);
      }

      await env.LEADERBOARD_KV.put(key, JSON.stringify(entries));
      return json({ ok: true, board, id: entry.id, count: entries.length });
    }

    return json({ error: "method not allowed" }, 405);
  },
};
