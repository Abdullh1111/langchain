// [T1] createAgent — hate-lekha while loop-er official replacement.
//
// Purono index.js (~30 line loop: invoke → tool_calls check → execute →
// ToolMessage push → repeat) ei ekta call-e dhuke geche, ar shathe middleware,
// structured output, streaming, budget shob free te ashche.
import { createAgent } from "langchain";
import * as z from "zod";

import { makeModel } from "./model.js";
import { buildMiddleware } from "./middleware.js";
import { buildKnowledgeTool } from "./knowledge.js";
import { buildSubagentTool } from "./subagent.js";
import { loadMcpTools } from "./mcp.js";
import { staticTools } from "./tools.js";
import { replyFormat, forcedToolFormat } from "./schema.js";

const SYSTEM_PROMPT = `You are an internal company assistant.

Rules:
- For any question about company policy (leave, expenses, deploys), you MUST call
  search_handbook first. Never answer policy questions from memory.
- If the handbook does not cover it, say so plainly instead of guessing.
- Use the task tool to delegate multi-step research or rewriting to a subagent.
- If you cannot resolve something, call escalate_to_human.
- Be concise. No preamble, no "Great question!".`;

// [T5] contextSchema — per-run data. Tool ar middleware ei ta
// `runtime.context` diye pore. Ekhane plain zod use korchi, LangGraph-er
// StateSchema na — tai eta LangGraph-free.
export const contextSchema = z.object({
  userName: z.string(),
  role: z.enum(["staff", "admin"]),
  timezone: z.string().default("Australia/Sydney"),
});

export async function buildAgents({ verbose = true } = {}) {
  const notes = [];

  const { tool: knowledgeTool, chunks } = await buildKnowledgeTool();
  notes.push(`handbook indexed: ${chunks} chunks`);

  const { tools: mcpTools, reason } = await loadMcpTools();
  notes.push(mcpTools.length ? `mcp tools: ${mcpTools.length}` : `mcp off (${reason})`);

  const taskTool = buildSubagentTool({ knowledgeTool });

  const tools = [...staticTools, knowledgeTool, taskTool, ...mcpTools];
  const middleware = buildMiddleware({ verbose });
  const model = makeModel();

  const shared = { model, tools, middleware, systemPrompt: SYSTEM_PROMPT, contextSchema };

  return {
    // Streaming ar token-by-token output er jonno — responseFormat nei.
    chat: createAgent({ ...shared, name: "assistant" }),

    // [T3] Structured output er jonno alada instance. responseFormat set thakle
    // final message ta structured-output tool call, tai text stream faka thake —
    // ei duita use-case ke alada rakhai porishkar.
    //
    // FORCE_TOOL_STRATEGY=1 dile toolStrategy + handleError path ta cholbe.
    structured: createAgent({
      ...shared,
      name: "assistant-structured",
      responseFormat:
        process.env.FORCE_TOOL_STRATEGY === "1" ? forcedToolFormat : replyFormat,
    }),

    tools,
    middlewareCount: middleware.length,
    notes,
  };
}
