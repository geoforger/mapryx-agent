import { createWorkersAI } from "workers-ai-provider";
import {
  callable,
  routeAgentRequest,
  type Connection,
  type ConnectionContext,
  type Schedule
} from "agents";
import { scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage,
  type UIMessage
} from "ai";
import { jwtVerify, type JWTPayload } from "jose";
import { z } from "zod";
import { CreditService } from "./credits";

// ---------------------------------------------------------------------------
// Token estimation helpers
// ---------------------------------------------------------------------------

/**
 * Rough token estimate: ~4 characters per token (GPT/Claude heuristic).
 * Good enough for logging; replace with a real tiktoken call if precision matters.
 */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Serialise any value to a string for token estimation. */
function stringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return String(value);
  }
}

/** Estimate tokens for an array of model messages (role + content). */
function estimateMessagesTokens(messages: ModelMessage[]): number {
  return messages.reduce((total, msg) => {
    const roleTokens = estimateTokens(msg.role);
    const contentTokens =
      typeof msg.content === "string"
        ? estimateTokens(msg.content)
        : Array.isArray(msg.content)
          ? (msg.content as Array<{ type?: string; text?: string }>).reduce(
              (sum, part) => sum + estimateTokens(stringify(part)),
              0
            )
          : estimateTokens(stringify(msg.content));
    return total + roleTokens + contentTokens;
  }, 0);
}

// ---------------------------------------------------------------------------
// Data-URI inlining (existing)
// ---------------------------------------------------------------------------

/**
 * The AI SDK's downloadAssets step runs `new URL(data)` on every file
 * part's string data. Data URIs parse as valid URLs, so it tries to
 * HTTP-fetch them and fails. Decode to Uint8Array so the SDK treats
 * them as inline data instead.
 */
function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((msg) => {
    if (msg.role !== "user" || typeof msg.content === "string") return msg;
    return {
      ...msg,
      content: msg.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;
        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;
        const bytes = Uint8Array.from(atob(match[2]), (c) => c.charCodeAt(0));
        return { ...part, data: bytes, mediaType: match[1] };
      })
    };
  });
}

/**
 * Count the number of tool invocations already made in the current agent turn
 * (since the last user message). Used to enforce a cumulative tool-call cap
 * across continuation turns, because stepCountIs() resets on every
 * onChatMessage() invocation.
 */
function countToolCallsInCurrentTurn(messages: UIMessage[]): number {
  let count = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user") break;
    if (msg.role === "assistant") {
      for (const part of msg.parts ?? []) {
        const compact = part as { type?: string; toolCallId?: string };
        if (
          compact.type === "tool-invocation" ||
          (compact.type?.startsWith("tool-") && compact.toolCallId)
        ) {
          count++;
        }
      }
    }
  }
  return count;
}

const MAPRYX_CLIENT_TOOL_GUIDE = [
  {
    name: "run_spatial_query",
    category: "sql",
    summary: "Execute DuckDB-WASM spatial SQL and create a result layer.",
    keywords: ["sql", "query", "duckdb", "st_", "spatial query", "custom"]
  },
  {
    name: "select_layer",
    category: "selection",
    summary: "Create a layer from features matching a SQL WHERE filter.",
    keywords: ["select", "where", "filter features", "subset"]
  },
  {
    name: "buffer_layer",
    category: "geoprocessing",
    summary: "Buffer layer features by distance and units.",
    keywords: ["buffer", "radius", "distance", "near", "around", "within"]
  },
  {
    name: "clip_layer",
    category: "overlay",
    summary: "Clip one layer to the boundary of a mask layer.",
    keywords: ["clip", "mask", "boundary", "cut"]
  },
  {
    name: "intersect_layers",
    category: "overlay",
    summary: "Create geometries where two layers overlap.",
    keywords: ["intersect", "intersection", "overlap", "common area"]
  },
  {
    name: "dissolve_layer",
    category: "geoprocessing",
    summary: "Dissolve features, optionally grouped by an attribute.",
    keywords: ["dissolve", "merge by", "aggregate", "group"]
  },
  {
    name: "split_layer",
    category: "data management",
    summary: "Split one layer into layers by unique attribute values.",
    keywords: ["split", "separate", "unique values", "by attribute"]
  },
  {
    name: "union_layer",
    category: "geoprocessing",
    summary: "Union all features in one layer into a single geometry.",
    keywords: ["union", "combine geometries", "single geometry"]
  },
  {
    name: "attribute_join_layers",
    category: "data management",
    summary: "Join two layers/tables on matching attribute values.",
    keywords: ["attribute join", "join by", "left key", "right key"]
  },
  {
    name: "spatial_join_layers",
    category: "overlay",
    summary: "Join attributes between layers by spatial relationship.",
    keywords: ["spatial join", "join", "contains", "within", "intersects"]
  },
  {
    name: "calculate_geometry",
    category: "analysis",
    summary: "Calculate area, length, or perimeter fields.",
    keywords: ["area", "length", "perimeter", "geometry property", "measure"]
  },
  {
    name: "calculate_field",
    category: "data management",
    summary: "Add or update a column from a DuckDB SQL expression.",
    keywords: ["calculate field", "new column", "expression", "derive"]
  },
  {
    name: "categorize_layer",
    category: "styling",
    summary: "Apply categorical symbology from an attribute.",
    keywords: ["categorize", "category", "symbology", "color by", "style by"]
  },
  {
    name: "filter_layer",
    category: "styling",
    summary: "Apply a visual attribute filter to a layer.",
    keywords: [
      "visual filter",
      "hide by filter",
      "filter layer",
      "show",
      "over",
      "under",
      "greater than",
      "less than",
      "elevation"
    ]
  },
  {
    name: "toggle_category_visibility",
    category: "styling",
    summary: "Show or hide one category value in a categorized layer.",
    keywords: ["show category", "hide category", "toggle category"]
  },
  {
    name: "update_layer_style",
    category: "styling",
    summary: "Change color, opacity, line width, or point radius.",
    keywords: ["style", "color", "opacity", "line width", "point radius"]
  },
  {
    name: "list_layers",
    category: "inspection",
    summary: "List loaded layers, optionally with attributes.",
    keywords: ["layers", "list layers", "loaded", "visible"]
  },
  {
    name: "inspect_layer",
    category: "inspection",
    summary: "Inspect schema and sample data for one layer.",
    keywords: ["inspect", "schema", "columns", "sample", "table name"]
  },
  {
    name: "get_unique_values",
    category: "inspection",
    summary: "Fetch unique values for a layer attribute.",
    keywords: ["unique values", "distinct", "values", "categories"]
  },
  {
    name: "query_spatial_functions",
    category: "inspection",
    summary: "Search available DuckDB spatial functions.",
    keywords: ["spatial functions", "function", "st_", "duckdb functions"]
  },
  {
    name: "points_to_path",
    category: "data management",
    summary: "Connect point features into lines or polygons.",
    keywords: ["points to path", "line from points", "track", "route"]
  },
  {
    name: "extract_mvt_data",
    category: "data management",
    summary: "Extract remote MVT features into a local layer.",
    keywords: ["mvt", "vector tile", "extract tile", "remote layer"]
  },
  {
    name: "merge_layers",
    category: "data management",
    summary: "Merge multiple layers into one layer.",
    keywords: ["merge layers", "append", "combine layers"]
  },
  {
    name: "erase_layer",
    category: "overlay",
    summary: "Erase input features that overlap an erase layer.",
    keywords: ["erase", "difference", "remove overlap", "subtract"]
  },
  {
    name: "check_system_status",
    category: "inspection",
    summary: "Check whether DuckDB spatial support is working.",
    keywords: ["status", "system", "duckdb spatial", "health"]
  }
] as const;

type MapryxClientToolName = (typeof MAPRYX_CLIENT_TOOL_GUIDE)[number]["name"];

const DEFAULT_CLIENT_TOOLS: MapryxClientToolName[] = [
  "list_layers",
  "inspect_layer"
];
const GENERIC_TOOL_TERMS = new Set([
  "layer",
  "layers",
  "spatial",
  "tool",
  "tools"
]);

function getToolSelectionText(messages: UIMessage[]): string {
  const recent = messages.slice(-8);
  return recent
    .map((message) =>
      stringify({
        role: message.role,
        parts: message.parts?.map((part) => {
          const compact = part as {
            type?: string;
            text?: string;
            toolName?: string;
            output?: unknown;
          };
          if (compact.type === "text") return compact.text;
          if (compact.toolName) return compact.toolName;
          if (compact.output) return stringify(compact.output).slice(0, 1200);
          return compact.type;
        })
      })
    )
    .join("\n")
    .toLowerCase();
}

function scoreMapryxTool(
  toolInfo: (typeof MAPRYX_CLIENT_TOOL_GUIDE)[number],
  query: string
): number {
  let score = 0;
  if (query.includes(toolInfo.name.toLowerCase())) score += 8;
  if (query.includes(toolInfo.category)) score += 2;
  for (const keyword of toolInfo.keywords) {
    if (query.includes(keyword.toLowerCase())) score += 3;
  }
  for (const term of toolInfo.name.split("_")) {
    if (
      term.length > 3 &&
      !GENERIC_TOOL_TERMS.has(term) &&
      query.includes(term)
    ) {
      score += 1;
    }
  }
  return score;
}

function searchMapryxClientTools(query: string, limit = 6) {
  const normalizedQuery = query.toLowerCase();
  return MAPRYX_CLIENT_TOOL_GUIDE.map((toolInfo) => ({
    ...toolInfo,
    score: scoreMapryxTool(toolInfo, normalizedQuery)
  }))
    .filter((toolInfo) => toolInfo.score > 0)
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, limit)
    .map(({ score: _score, keywords: _keywords, ...toolInfo }) => toolInfo);
}

function selectMapryxClientTools(
  messages: UIMessage[]
): MapryxClientToolName[] {
  const searchText = getToolSelectionText(messages);
  const selected = new Set<MapryxClientToolName>(DEFAULT_CLIENT_TOOLS);

  for (const result of searchMapryxClientTools(searchText, 8)) {
    selected.add(result.name);
  }

  if (
    /\bst_|sql|duckdb|query|where|join|buffer|clip|intersect/.test(searchText)
  ) {
    selected.add("run_spatial_query");
  }

  return [...selected].slice(0, 10);
}

function shouldExposeScheduleTools(messages: UIMessage[]): boolean {
  return /\bschedule|scheduled|remind|reminder|later|tomorrow|cron|daily|weekly|monthly|cancel.+task|tasks?\b/.test(
    getToolSelectionText(messages)
  );
}

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  private async validateSessionToken(sessionToken: string | undefined) {
    if (!sessionToken) {
      throw new Error("session_token_required");
    }
    if (!this.env.AGENT_SESSION_SECRET) {
      throw new Error("agent_session_secret_required");
    }

    try {
      const { payload } = await jwtVerify(
        sessionToken,
        new TextEncoder().encode(this.env.AGENT_SESSION_SECRET),
        { algorithms: ["HS256"] }
      );
      this.setIdentityContext(payload);
    } catch {
      throw new Error("invalid_session_token");
    }
  }

  private setIdentityContext(payload: JWTPayload) {
    const target = this as unknown as {
      ctx?: {
        userId?: string;
        planId?: string;
      };
    };
    target.ctx = target.ctx ?? {};
    target.ctx.userId = payload.sub;
    target.ctx.planId =
      typeof payload.planId === "string" ? payload.planId : undefined;
  }

  async onConnect(_connection: Connection, context: ConnectionContext) {
    const url = new URL(context.request.url);
    const headerToken = context.request.headers.get("X-Agent-Session-Token");
    const queryToken = url.searchParams.get("sessionToken");
    let bodyToken: string | undefined;

    try {
      const body = (await context.request.clone().json()) as {
        sessionToken?: string;
      };
      bodyToken = body.sessionToken;
    } catch {
      // WebSocket upgrade requests generally have no JSON body.
    }

    await this.validateSessionToken(
      bodyToken ?? headerToken ?? queryToken ?? undefined
    );
  }

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const body = options?.body as
      | { systemPrompt?: string; sessionToken?: string }
      | undefined;
    await this.validateSessionToken(body?.sessionToken);

    const target = this as unknown as { ctx?: { userId?: string } };
    const userId = target.ctx?.userId;

    let creditService: CreditService | null = null;
    let agentRunId: string | null = null;

    if (userId && this.env.SUPABASE_URL && this.env.SUPABASE_SERVICE_ROLE_KEY) {
      creditService = new CreditService(
        this.env.SUPABASE_URL,
        this.env.SUPABASE_SERVICE_ROLE_KEY
      );
      agentRunId = crypto.randomUUID();
      const reservation = await creditService.reserveCredits(
        userId,
        agentRunId
      );
      if (!reservation.success) {
        return new Response(
          "Error: Insufficient credits. Your available balance is too low. Please top up your credits in the billing panel."
        );
      }
    }

    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    // Accept system prompt injected by the Mapryx client (includes live schema context)
    const systemPrompt =
      body?.systemPrompt ??
      `You are Mapryx AI Agent, a geospatial analysis assistant built into the Mapryx platform.

You help users analyze spatial data, explore layers, run geospatial operations, and understand their geographic datasets. Be concise and technical.

Most tools (DuckDB queries, layer management, spatial joins, buffer/clip/dissolve, etc.) run directly in the user's browser. The server exposes only the browser tools likely needed for the current request. Use searchMapryxTools when you need to discover a tool before acting. Server-side tools: echo (debug), getMapryxCapabilities, searchMapryxTools, and task scheduling.
`;

    const modelId = "@cf/zai-org/glm-4.7-flash";

    // -----------------------------------------------------------------------
    // TOKEN LOGGING — Step 1: raw client message
    // -----------------------------------------------------------------------
    const lastClientMessage = this.messages[this.messages.length - 1];
    // UIMessage uses `parts` (not `content`) — serialise the whole message for logging
    const rawClientText = lastClientMessage
      ? stringify(
          (
            lastClientMessage as unknown as {
              parts?: unknown;
              content?: unknown;
            }
          ).parts ?? lastClientMessage
        )
      : "";
    const clientTokensEst = estimateTokens(rawClientText);

    console.log(
      "\n╔══════════════════════════════════════════════════════════════╗"
    );
    console.log("║  [token-log] INCOMING CLIENT MESSAGE");
    console.log(
      "╚══════════════════════════════════════════════════════════════╝"
    );
    console.log(`[token-log] userId        : ${userId ?? "(anonymous)"}`);
    console.log(
      `[token-log] agentRunId    : ${agentRunId ?? "(no credit run)"}`
    );
    console.log(`[token-log] model         : ${modelId}`);
    console.log(
      `[token-log] client msgs   : ${this.messages.length} stored message(s)`
    );
    console.log(
      `[token-log] last role     : ${lastClientMessage?.role ?? "n/a"}`
    );
    console.log(
      `[token-log] ~client tokens: ${clientTokensEst} (last message)`
    );
    console.log("[token-log] FULL last message content:");
    console.log(rawClientText);

    // -----------------------------------------------------------------------
    // TOKEN LOGGING — Step 2: build the final messages array (same as LLM call)
    // -----------------------------------------------------------------------
    const prunedMessages = pruneMessages({
      messages: inlineDataUrls(await convertToModelMessages(this.messages)),
      toolCalls: "before-last-2-messages"
    });

    const systemPromptTokensEst = estimateTokens(systemPrompt);
    const historyTokensEst = estimateMessagesTokens(prunedMessages);
    const totalInputTokensEst = systemPromptTokensEst + historyTokensEst;

    console.log(
      "\n╔══════════════════════════════════════════════════════════════╗"
    );
    console.log("║  [token-log] BACKEND ADDITIONS");
    console.log(
      "╚══════════════════════════════════════════════════════════════╝"
    );
    console.log(`[token-log] ~system prompt tokens : ${systemPromptTokensEst}`);
    console.log(
      `[token-log] system prompt source  : ${body?.systemPrompt ? "client-injected" : "default fallback"}`
    );
    console.log(
      `[token-log] history messages count: ${prunedMessages.length} (after pruning)`
    );
    console.log(`[token-log] ~history tokens       : ${historyTokensEst}`);
    console.log(
      "\n╔══════════════════════════════════════════════════════════════╗"
    );
    console.log("║  [token-log] TOTAL TOKENS GOING TO LLM (estimated)");
    console.log(
      "╚══════════════════════════════════════════════════════════════╝"
    );
    console.log(`[token-log] ~system prompt tokens : ${systemPromptTokensEst}`);
    console.log(`[token-log] ~history tokens       : ${historyTokensEst}`);
    console.log(`[token-log] ─────────────────────────────────────────`);
    console.log(`[token-log] ~TOTAL input tokens   : ${totalInputTokensEst}`);
    console.log("[token-log] FULL system prompt:");
    console.log(systemPrompt);
    console.log("[token-log] FULL pruned messages going to LLM:");
    console.log(JSON.stringify(prunedMessages, null, 2));

    // Compute remaining budget relative to tool calls already made this turn.
    // This prevents the infinite-loop bug where stepCountIs resets on every
    // continuation (each client tool result triggers a new onChatMessage call).
    const MAX_TOOL_CALLS_PER_TURN = 10;
    const toolCallsThisTurn = countToolCallsInCurrentTurn(
      this.messages as unknown as UIMessage[]
    );
    const remainingSteps = Math.max(
      1,
      MAX_TOOL_CALLS_PER_TURN - toolCallsThisTurn
    );

    console.log(
      `[step-guard] tool calls this turn: ${toolCallsThisTurn}, remaining steps: ${remainingSteps}`
    );

    const selectedClientToolNames = selectMapryxClientTools(
      this.messages as unknown as UIMessage[]
    );
    const exposeScheduleTools = shouldExposeScheduleTools(
      this.messages as unknown as UIMessage[]
    );
    console.log(
      `[tool-selection] client tools exposed: ${selectedClientToolNames.join(", ")}`
    );
    console.log(
      `[tool-selection] schedule tools exposed: ${exposeScheduleTools ? "yes" : "no"}`
    );

    let result;
    try {
      result = streamText({
        model: workersai(modelId, {
          sessionAffinity: this.sessionAffinity
        }),
        system: systemPrompt,
        // Use the already-built prunedMessages (also logged above)
        messages: prunedMessages,
        tools: {
          // MCP tools from connected servers
          ...mcpTools,

          echo: tool({
            description:
              "Echo back a message — useful for testing the agent pipeline",
            inputSchema: z.object({
              message: z.string().describe("Message to echo back")
            }),
            execute: async ({ message }) => ({
              echo: message,
              timestamp: new Date().toISOString()
            })
          }),

          getMapryxCapabilities: tool({
            description:
              "Get a compact categorized overview of available Mapryx geospatial capabilities. Use searchMapryxTools for detailed tool matches.",
            inputSchema: z.object({}),
            execute: async () => {
              const categories = MAPRYX_CLIENT_TOOL_GUIDE.reduce<
                Record<string, string[]>
              >((acc, toolInfo) => {
                acc[toolInfo.category] = acc[toolInfo.category] ?? [];
                acc[toolInfo.category].push(toolInfo.name);
                return acc;
              }, {});

              return {
                serverTools: [
                  "echo",
                  "getMapryxCapabilities",
                  "searchMapryxTools",
                  ...(exposeScheduleTools
                    ? [
                        "scheduleTask",
                        "getScheduledTasks",
                        "cancelScheduledTask"
                      ]
                    : [])
                ],
                browserToolCategories: categories,
                activeBrowserTools: selectedClientToolNames,
                guidance:
                  "Browser tools execute in the Mapryx client. Call searchMapryxTools with the user's intent when the needed exact tool is not currently active."
              };
            }
          }),

          searchMapryxTools: tool({
            description:
              "Search Mapryx browser-executed geospatial tools by intent and return compact matching tool names and summaries.",
            inputSchema: z.object({
              query: z
                .string()
                .describe("User intent or operation to search for"),
              limit: z
                .number()
                .min(1)
                .max(10)
                .optional()
                .describe("Maximum number of matching tools to return")
            }),
            execute: async ({ query, limit }) => ({
              matches: searchMapryxClientTools(query, limit ?? 6),
              activeBrowserTools: selectedClientToolNames,
              note: "If a matching tool is not active, ask the user to restate the operation with the tool name or continue with an active equivalent such as run_spatial_query when present."
            })
          }),

          ...(exposeScheduleTools
            ? {
                scheduleTask: tool({
                  description:
                    "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
                  inputSchema: scheduleSchema,
                  execute: async ({ when, description }) => {
                    if (when.type === "no-schedule") {
                      return "Not a valid schedule input";
                    }
                    const input =
                      when.type === "scheduled"
                        ? when.date
                        : when.type === "delayed"
                          ? when.delayInSeconds
                          : when.type === "cron"
                            ? when.cron
                            : null;
                    if (!input) return "Invalid schedule type";
                    try {
                      this.schedule(input, "executeTask", description, {
                        idempotent: true
                      });
                      return `Task scheduled: "${description}" (${when.type}: ${input})`;
                    } catch (error) {
                      return `Error scheduling task: ${error}`;
                    }
                  }
                }),

                getScheduledTasks: tool({
                  description: "List all tasks that have been scheduled",
                  inputSchema: z.object({}),
                  execute: async () => {
                    const tasks = this.getSchedules();
                    return tasks.length > 0
                      ? tasks
                      : "No scheduled tasks found.";
                  }
                }),

                cancelScheduledTask: tool({
                  description: "Cancel a scheduled task by its ID",
                  inputSchema: z.object({
                    taskId: z.string().describe("The ID of the task to cancel")
                  }),
                  execute: async ({ taskId }) => {
                    try {
                      this.cancelSchedule(taskId);
                      return `Task ${taskId} cancelled.`;
                    } catch (error) {
                      return `Error cancelling task: ${error}`;
                    }
                  }
                })
              }
            : {}),

          // --- Mapryx spatial tools — executed in the browser via onToolCall ---
          ...(selectedClientToolNames.includes("run_spatial_query")
            ? {
                run_spatial_query: tool({
                  description:
                    "Execute a DuckDB-WASM spatial SQL query and add the result as a new layer",
                  inputSchema: z.object({
                    query: z.string().describe("SQL query to execute"),
                    description: z
                      .string()
                      .describe(
                        "Human-readable description of what this query does"
                      )
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("select_layer")
            ? {
                select_layer: tool({
                  description:
                    "Filter features from a layer using a SQL WHERE condition",
                  inputSchema: z.object({
                    layerName: z.string().describe("Source layer name"),
                    filter: z.string().optional().describe("SQL WHERE clause"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("buffer_layer")
            ? {
                buffer_layer: tool({
                  description:
                    "Create a buffer around each feature by distance and units",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to buffer"),
                    distance: z.number().describe("Buffer distance"),
                    units: z
                      .enum(["meters", "kilometers", "miles", "feet"])
                      .describe("Distance units"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("clip_layer")
            ? {
                clip_layer: tool({
                  description: "Clip a layer to the boundary of a mask layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to clip"),
                    maskLayerName: z.string().describe("Mask layer"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("intersect_layers")
            ? {
                intersect_layers: tool({
                  description:
                    "Compute the geometric intersection of two layers",
                  inputSchema: z.object({
                    layerName: z.string().describe("First layer"),
                    targetLayerName: z.string().describe("Second layer"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("dissolve_layer")
            ? {
                dissolve_layer: tool({
                  description:
                    "Dissolve features in a layer, optionally grouped by an attribute",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to dissolve"),
                    groupByAttribute: z
                      .string()
                      .optional()
                      .describe("Attribute to group by"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("split_layer")
            ? {
                split_layer: tool({
                  description:
                    "Split a layer into multiple layers based on unique values of an attribute",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to split"),
                    splitByAttribute: z
                      .string()
                      .describe("Attribute to split by"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Base name for output layers")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("union_layer")
            ? {
                union_layer: tool({
                  description:
                    "Union all features in a layer into a single geometry",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to union"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("attribute_join_layers")
            ? {
                attribute_join_layers: tool({
                  description: "Join two layers on matching attribute values",
                  inputSchema: z.object({
                    leftLayerName: z.string().describe("Left layer"),
                    rightLayerName: z.string().describe("Right layer"),
                    leftKey: z.string().describe("Join key in left layer"),
                    rightKey: z.string().describe("Join key in right layer"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("spatial_join_layers")
            ? {
                spatial_join_layers: tool({
                  description: "Join two layers based on spatial relationship",
                  inputSchema: z.object({
                    leftLayerName: z.string().describe("Base layer"),
                    rightLayerName: z.string().describe("Layer to join"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("calculate_geometry")
            ? {
                calculate_geometry: tool({
                  description:
                    "Calculate a geometry property (area, length, perimeter) for each feature",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    property: z
                      .enum(["area", "length", "perimeter"])
                      .describe("Property to calculate"),
                    unit: z.string().optional().describe("Unit of measurement"),
                    srid: z
                      .string()
                      .optional()
                      .describe("Projection SRID (e.g. EPSG:3857)"),
                    decimalPlaces: z
                      .number()
                      .optional()
                      .describe("Decimal places for result")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("calculate_field")
            ? {
                calculate_field: tool({
                  description: "Add or update a column using a SQL expression",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    columnName: z
                      .string()
                      .describe("Column name to create or update"),
                    dataType: z
                      .string()
                      .describe("SQL data type (e.g. DOUBLE, VARCHAR)"),
                    expression: z
                      .string()
                      .describe("SQL expression for the value")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("categorize_layer")
            ? {
                categorize_layer: tool({
                  description:
                    "Apply categorical symbology to a layer based on an attribute",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    attributeName: z
                      .string()
                      .describe("Attribute to categorize by")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("filter_layer")
            ? {
                filter_layer: tool({
                  description:
                    "Apply a visual filter to a layer based on attribute conditions",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    filters: z
                      .array(
                        z.object({
                          attribute: z.string(),
                          operator: z.string(),
                          value: z.union([z.string(), z.number()])
                        })
                      )
                      .describe("Filter conditions")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("toggle_category_visibility")
            ? {
                toggle_category_visibility: tool({
                  description:
                    "Show or hide a specific category value in a categorized layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    categoryValue: z
                      .string()
                      .describe("Category value to toggle"),
                    action: z.enum(["show", "hide"]).describe("Action to take")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("update_layer_style")
            ? {
                update_layer_style: tool({
                  description: "Update the visual style of a layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    style: z
                      .object({
                        color: z.string().optional(),
                        opacity: z.number().optional(),
                        lineWidth: z.number().optional(),
                        pointRadius: z.number().optional()
                      })
                      .describe("Style properties to update")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("list_layers")
            ? {
                list_layers: tool({
                  description:
                    "List all loaded layers with optional attribute details",
                  inputSchema: z.object({
                    includeAttributes: z
                      .boolean()
                      .optional()
                      .describe("Include attribute names"),
                    visibleOnly: z
                      .boolean()
                      .optional()
                      .describe("Only return visible layers")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("inspect_layer")
            ? {
                inspect_layer: tool({
                  description:
                    "Get detailed schema and sample data from a layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Layer to inspect")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("get_unique_values")
            ? {
                get_unique_values: tool({
                  description: "Get unique values for an attribute in a layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Target layer"),
                    attributeName: z
                      .string()
                      .describe("Attribute to get unique values for"),
                    limit: z
                      .number()
                      .optional()
                      .describe("Maximum number of values to return")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("query_spatial_functions")
            ? {
                query_spatial_functions: tool({
                  description:
                    "Search for available DuckDB spatial functions matching a pattern",
                  inputSchema: z.object({
                    functionPattern: z
                      .string()
                      .describe("Pattern to search for (e.g. ST_Buffer)")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("points_to_path")
            ? {
                points_to_path: tool({
                  description:
                    "Connect point features into a path/line based on an ordering attribute",
                  inputSchema: z.object({
                    layerName: z.string().describe("Point layer to convert"),
                    orderByAttribute: z
                      .string()
                      .optional()
                      .describe("Attribute to order points by"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the output layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("extract_mvt_data")
            ? {
                extract_mvt_data: tool({
                  description:
                    "Extract features from a remote MVT layer into a local manageable layer by fetching and merging tiles",
                  inputSchema: z.object({
                    layerId: z
                      .string()
                      .describe("ID of the MVT layer to extract from"),
                    zoom: z.number().describe("Zoom level to extract at"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the resulting vector layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("merge_layers")
            ? {
                merge_layers: tool({
                  description:
                    "Combine multiple layers into a single new layer with merged attributes",
                  inputSchema: z.object({
                    layerNames: z
                      .array(z.string())
                      .describe("Names or partial names of layers to merge"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the resulting layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("erase_layer")
            ? {
                erase_layer: tool({
                  description:
                    "Erase features in the input layer that overlap with features in the erase layer",
                  inputSchema: z.object({
                    layerName: z.string().describe("Input layer to erase from"),
                    eraseLayerName: z
                      .string()
                      .describe("Layer used for erasing"),
                    outputName: z
                      .string()
                      .optional()
                      .describe("Name for the resulting layer")
                  })
                })
              }
            : {}),

          ...(selectedClientToolNames.includes("check_system_status")
            ? {
                check_system_status: tool({
                  description:
                    "Check if the DuckDB spatial extension is loaded and working",
                  inputSchema: z.object({})
                })
              }
            : {})
        },
        stopWhen: stepCountIs(remainingSteps),
        abortSignal: options?.abortSignal
      });
    } catch (error) {
      if (creditService && userId && agentRunId) {
        await creditService.releaseReservation(userId, agentRunId);
      }
      throw error;
    }

    // -----------------------------------------------------------------------
    // TOKEN LOGGING — Step 3: actual LLM usage after streaming completes
    // -----------------------------------------------------------------------
    this.ctx.waitUntil(
      // Wrap in Promise.resolve() because result.usage is PromiseLike (no .catch)
      Promise.resolve(result.usage)
        .then((usage) => {
          console.log(
            "\n╔══════════════════════════════════════════════════════════════╗"
          );
          console.log(
            "║  [token-log] ACTUAL LLM TOKEN USAGE (reported by API)"
          );
          console.log(
            "╚══════════════════════════════════════════════════════════════╝"
          );
          console.log(
            `[token-log] input tokens  (actual): ${usage.inputTokens ?? "n/a"}`
          );
          console.log(
            `[token-log] output tokens (actual): ${usage.outputTokens ?? "n/a"}`
          );
          console.log(
            `[token-log] total tokens  (actual): ${(usage.inputTokens ?? 0) + (usage.outputTokens ?? 0)}`
          );
          console.log(`[token-log] userId        : ${userId ?? "(anonymous)"}`);
          console.log(`[token-log] model         : ${modelId}`);
        })
        .catch((err: unknown) => {
          console.warn("[token-log] could not read usage:", err);
        })
    );

    if (creditService && userId && agentRunId) {
      const settle = async () => {
        try {
          const usage = await result.usage;
          await creditService!.settleUsage(
            userId!,
            agentRunId!,
            usage.inputTokens ?? 0,
            usage.outputTokens ?? 0,
            modelId
          );
        } catch (err) {
          console.error("[credits] settlement error:", err);
        }
      };
      this.ctx.waitUntil(settle());
    }

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

const ALLOWED_ORIGINS = [
  "https://mapryx-staging.geoforger.com",
  "https://mapryx.geoforger.com",
  "http://localhost:5173",
  "http://localhost:5174"
];

const SESSION_COOKIE = "mapryx_agent_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 7;
const encoder = new TextEncoder();

type AgentEnv = Env & {
  ASSETS?: Fetcher;
  AGENT_ACCESS_PASSWORD?: string;
  AGENT_AUTH_TOKEN?: string;
  AGENT_SESSION_SECRET?: string;
};

function corsHeaders(origin: string | null): HeadersInit {
  const allowed =
    origin && ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    "Access-Control-Allow-Origin": allowed,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers":
      "Authorization, X-Agent-Auth-Token, Content-Type, Upgrade, Connection, Sec-WebSocket-Key, Sec-WebSocket-Version, Sec-WebSocket-Extensions, Sec-WebSocket-Protocol",
    "Access-Control-Allow-Credentials": "true"
  };
}

function withCors(response: Response, headers: HeadersInit): Response {
  const next = new Response(response.body, response);
  Object.entries(headers).forEach(([key, value]) =>
    next.headers.set(key, value)
  );
  return next;
}

function hasAuthConfig(env: AgentEnv): boolean {
  return Boolean(
    env.AGENT_ACCESS_PASSWORD ||
    env.AGENT_AUTH_TOKEN ||
    env.AGENT_SESSION_SECRET
  );
}

function getSessionSecret(env: AgentEnv): string | undefined {
  return (
    env.AGENT_SESSION_SECRET ||
    env.AGENT_AUTH_TOKEN ||
    env.AGENT_ACCESS_PASSWORD
  );
}

function parseCookies(header: string | null): Record<string, string> {
  if (!header) return {};
  const cookies: Record<string, string> = {};
  for (const part of header.split(";")) {
    const [name, ...value] = part.trim().split("=");
    if (name && value.length > 0) cookies[name] = value.join("=");
  }
  return cookies;
}

function base64Url(bytes: Uint8Array): string {
  let value = "";
  for (const byte of bytes) value += String.fromCharCode(byte);
  return btoa(value).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sha256(value: string): Promise<Uint8Array> {
  return new Uint8Array(
    await crypto.subtle.digest("SHA-256", encoder.encode(value))
  );
}

function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

async function secureStringEqual(a: string, b: string): Promise<boolean> {
  return constantTimeEqual(await sha256(a), await sha256(b));
}

async function signSession(secret: string, payload: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(payload)
  );
  return base64Url(new Uint8Array(signature));
}

async function createSessionCookie(
  env: AgentEnv,
  request: Request
): Promise<string> {
  const secret = getSessionSecret(env);
  if (!secret) throw new Error("Auth session secret is not configured");

  const expires = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const payload = String(expires);
  const signature = await signSession(secret, payload);
  const url = new URL(request.url);
  const secure = url.protocol === "https:" ? "; Secure" : "";

  return [
    `${SESSION_COOKIE}=${payload}.${signature}`,
    "Path=/",
    "HttpOnly",
    "SameSite=Lax",
    `Max-Age=${SESSION_TTL_SECONDS}`,
    secure
  ]
    .filter(Boolean)
    .join("; ");
}

async function verifySessionCookie(
  env: AgentEnv,
  cookieValue: string | undefined
): Promise<boolean> {
  const secret = getSessionSecret(env);
  if (!secret || !cookieValue) return false;

  const [expires, signature] = cookieValue.split(".");
  const expiresAt = Number(expires);
  if (!expires || !signature || !Number.isFinite(expiresAt)) return false;
  if (expiresAt < Math.floor(Date.now() / 1000)) return false;

  const expected = await signSession(secret, expires);
  return secureStringEqual(signature, expected);
}

function getBearerToken(request: Request): string | undefined {
  const authorization = request.headers.get("Authorization");
  if (authorization?.startsWith("Bearer "))
    return authorization.slice("Bearer ".length).trim();
  return request.headers.get("X-Agent-Auth-Token") ?? undefined;
}

async function authorizeRequest(
  request: Request,
  env: AgentEnv
): Promise<boolean> {
  if (
    env.AGENT_SESSION_SECRET &&
    !env.AGENT_ACCESS_PASSWORD &&
    !env.AGENT_AUTH_TOKEN
  ) {
    return true;
  }

  const token = getBearerToken(request);
  if (
    token &&
    env.AGENT_AUTH_TOKEN &&
    (await secureStringEqual(token, env.AGENT_AUTH_TOKEN))
  ) {
    return true;
  }

  const cookies = parseCookies(request.headers.get("Cookie"));
  return verifySessionCookie(env, cookies[SESSION_COOKIE]);
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function loginPage(nextPath: string, error?: string): Response {
  const safeNext =
    nextPath.startsWith("/") && !nextPath.startsWith("//") ? nextPath : "/";
  const errorHtml = error ? `<p class="error">${escapeHtml(error)}</p>` : "";

  return new Response(
    `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Mapryx Agent Sign In</title>
    <style>
      :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { min-height: 100vh; margin: 0; display: grid; place-items: center; background: #0f172a; color: #e5e7eb; }
      main { width: min(380px, calc(100vw - 32px)); }
      h1 { margin: 0 0 8px; font-size: 22px; line-height: 1.2; }
      p { margin: 0 0 22px; color: #9ca3af; font-size: 14px; }
      form { display: grid; gap: 12px; }
      label { display: grid; gap: 8px; font-size: 13px; color: #cbd5e1; }
      input { height: 42px; border: 1px solid #334155; border-radius: 8px; padding: 0 12px; background: #111827; color: #f8fafc; font-size: 15px; }
      button { height: 42px; border: 0; border-radius: 8px; background: #38bdf8; color: #082f49; font-weight: 700; cursor: pointer; }
      .error { color: #fca5a5; margin-bottom: 12px; }
    </style>
  </head>
  <body>
    <main>
      <h1>Mapryx Agent</h1>
      <p>Sign in to continue.</p>
      ${errorHtml}
      <form method="post" action="/auth/login">
        <input type="hidden" name="next" value="${escapeHtml(safeNext)}" />
        <label>
          Access password
          <input name="password" type="password" autocomplete="current-password" autofocus required />
        </label>
        <button type="submit">Sign in</button>
      </form>
    </main>
  </body>
</html>`,
    {
      status: error ? 401 : 200,
      headers: { "Content-Type": "text/html; charset=utf-8" }
    }
  );
}

async function handleLogin(request: Request, env: AgentEnv): Promise<Response> {
  const url = new URL(request.url);
  const nextPath = url.searchParams.get("next") || "/";

  if (!env.AGENT_ACCESS_PASSWORD) {
    return new Response("Browser login is not configured.", { status: 503 });
  }

  if (request.method === "GET") {
    return loginPage(nextPath);
  }

  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const form = await request.formData();
  const password = String(form.get("password") || "");
  const submittedNext = String(form.get("next") || "/");

  if (!(await secureStringEqual(password, env.AGENT_ACCESS_PASSWORD))) {
    return loginPage(submittedNext, "Invalid password.");
  }

  const safeNext =
    submittedNext.startsWith("/") && !submittedNext.startsWith("//")
      ? submittedNext
      : "/";
  return new Response(null, {
    status: 303,
    headers: {
      Location: safeNext,
      "Set-Cookie": await createSessionCookie(env, request)
    }
  });
}

function handleLogout(): Response {
  return new Response(null, {
    status: 303,
    headers: {
      Location: "/auth/login",
      "Set-Cookie": `${SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`
    }
  });
}

function unauthorizedResponse(request: Request, env: AgentEnv): Response {
  const url = new URL(request.url);
  const acceptsHtml =
    request.method === "GET" &&
    request.headers.get("Accept")?.includes("text/html") &&
    request.headers.get("Upgrade")?.toLowerCase() !== "websocket";

  if (!hasAuthConfig(env)) {
    return new Response(
      "Authentication is not configured for this deployment.",
      { status: 503 }
    );
  }

  if (acceptsHtml && env.AGENT_ACCESS_PASSWORD) {
    return new Response(null, {
      status: 303,
      headers: {
        Location: `/auth/login?next=${encodeURIComponent(url.pathname + url.search)}`
      }
    });
  }

  return new Response("Unauthorized", {
    status: 401,
    headers: { "WWW-Authenticate": "Bearer" }
  });
}

export default {
  async fetch(request: Request, env: AgentEnv) {
    const url = new URL(request.url);
    const origin = request.headers.get("Origin");
    const headers = corsHeaders(origin);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers });
    }

    if (url.pathname === "/auth/login") {
      return withCors(await handleLogin(request, env), headers);
    }

    if (url.pathname === "/auth/logout") {
      return withCors(handleLogout(), headers);
    }

    if (!(await authorizeRequest(request, env))) {
      return withCors(unauthorizedResponse(request, env), headers);
    }

    let req = request;
    const sessionToken = url.searchParams.get("sessionToken");
    if (sessionToken) {
      const newHeaders = new Headers(request.headers);
      newHeaders.set("X-Agent-Session-Token", sessionToken);
      req = new Request(request, { headers: newHeaders });
    }

    const response =
      (await routeAgentRequest(req, env)) ||
      (env.ASSETS
        ? await env.ASSETS.fetch(req)
        : new Response("Not found", { status: 404 }));

    return withCors(response, headers);
  }
} satisfies ExportedHandler<AgentEnv>;
