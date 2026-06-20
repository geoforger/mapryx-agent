# CLAUDE.md — mapryx-agent

This is a Cloudflare Workers AI agent project. It is a **sub-project** inside the main Mapryx monorepo. Run all commands from this directory (`mapryx-agent/`), not the repo root.

## Commands

```bash
npm run dev       # Vite dev server + wrangler local (http://localhost:5173)
npm run deploy    # vite build && wrangler deploy
npm run types     # Regenerate env.d.ts from wrangler.jsonc bindings
npm run lint      # oxlint src/
npm run check     # oxfmt check + oxlint + tsc
```

Run `npm run types` after any binding change in `wrangler.jsonc`.

## Architecture

- **`src/server.ts`** — `ChatAgent` Durable Object (extends `AIChatAgent`). Handles LLM calls via `workers-ai-provider`, MCP server management, and task scheduling via Durable Object alarms.
- **`src/app.tsx`** — React chat UI. Uses `useAgent` + `useAgentChat` from the agents SDK. WebSocket-connected to `ChatAgent`.
- **LLM**: `@cf/moonshotai/kimi-k2.6` via Workers AI binding (`env.AI`).
- **Persistence**: SQLite-backed Durable Object, up to 100 messages per session.

## Important Constraints

1. **`inlineDataUrls()` in `server.ts` must not be removed.** The AI SDK tries to HTTP-fetch data URIs; this function decodes them to `Uint8Array` first.
2. **Frontend tools** (`runDuckDBQuery`, `addLayer`, spatial ops) are injected by the Mapryx host app at runtime — they are not implemented here. Only declare them in `getMapryxCapabilities`.
3. **Cloudflare APIs change rapidly** — always fetch current docs before editing Workers/Agents/Durable Objects code.
4. **`stopWhen: stepCountIs(5)`** limits the agent loop to 5 steps per message. Increase deliberately.
5. **`pruneMessages`** strips old tool calls to save tokens. Keep this in `onChatMessage`.

## Linting

Uses `oxlint` (not ESLint) and `oxfmt` (not Prettier). Configuration in `.oxlintrc.json` and `.oxfmtrc.json`.

## Deployment

| Environment | Trigger                                                         |
| ----------- | --------------------------------------------------------------- |
| Local       | `npm run dev`                                                   |
| Production  | `npm run deploy` (Cloudflare Workers — name: `mapryx-agent-v2`) |

No CI/CD pipeline — deploy manually via `npm run deploy`.
