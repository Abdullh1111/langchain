// [T13] Subagents — LangChain only, no LangGraph.
//
// Pattern: ekta createAgent ke tool() diye wrap koro, tarpar supervisor agent
// er tools[] e dao. Eituku pura LangGraph-free.
//
// Ja ekhane INTENTIONALLY nei (LangGraph lage, alada repo-te):
//   - getCurrentTaskInput() diye parent state pora  → bodole arg hishebe pathaw
//   - new Command({ update }) diye parent state lekha → bodole string return koro
//   - subagent-er nijer checkpointer / interrupt
import { createAgent, tool } from "langchain";
import * as z from "zod";

import { makeModel, CHEAP_MODEL } from "./model.js";

/**
 * Subagent gula ke ekta registry-te rakhi ar EKTA `task` tool diye dispatch kori.
 * Keno: prottek agent-er jonno alada tool dile supervisor-er tool list phule
 * othe. Ekta dispatch tool = ekta tool slot, joto agent-i thakuk.
 */
export function buildSubagentTool({ knowledgeTool }) {
  const researcher = createAgent({
    name: "researcher",
    description: "Digs through the company handbook and reports findings.",
    model: makeModel({ model: CHEAP_MODEL }),
    tools: [knowledgeTool],
    systemPrompt:
      "You are a research subagent. Search the handbook and report ONLY what you " +
      "found, with the policy name. If the handbook does not cover it, say so plainly. " +
      "Never guess. Be terse — bullet points, no preamble.",
  });

  const writer = createAgent({
    name: "writer",
    description: "Turns rough notes into a short, clean message.",
    model: makeModel({ model: CHEAP_MODEL }),
    tools: [],
    systemPrompt:
      "You are a writing subagent. Rewrite the given notes as a clear, short reply. " +
      "Plain language, no marketing tone, no invented facts.",
  });

  const registry = { researcher, writer };

  const task = tool(
    async ({ agentName, description }) => {
      const agent = registry[agentName];
      if (!agent) {
        // Model ke bhul agent name er jonno correct list ta feedback dao.
        return `Unknown agent "${agentName}". Available: ${Object.keys(registry).join(", ")}.`;
      }

      const result = await agent.invoke({
        messages: [{ role: "user", content: description }],
      });

      // .text safer than .content — content string ba block-array dono hote pare.
      return result.messages.at(-1)?.text ?? "(subagent returned nothing)";
    },
    {
      name: "task",
      description: `Delegate a self-contained task to a subagent.

Available agents:
- researcher: search the company handbook and report findings
- writer: turn rough notes into a short clean message

The subagent sees ONLY your description — it cannot see this conversation.
So put every fact it needs into the description.`,
      schema: z.object({
        agentName: z.enum(["researcher", "writer"]),
        description: z.string().describe("Self-contained task description"),
      }),
    },
  );

  return task;
}
