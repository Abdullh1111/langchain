// [T6] Tools — advanced options + ToolRuntime.
import { tool, ToolMessage } from "langchain";
import * as z from "zod";

// ── plain tool ────────────────────────────────────────────────────────────────
export const getWeather = tool(
  async ({ city }) => `It's 24°C and sunny in ${city}.`,
  {
    name: "get_weather",
    description: "Get the current weather for a city.",
    schema: z.object({ city: z.string().describe("City name") }),
  },
);

// ── [T6] responseFormat: "content_and_artifact" ───────────────────────────────
// Function ke ekta 2-tuple return korte HOBE: [content, artifact].
//   content  → ToolMessage.content  → model ei ta dekhe (token khay)
//   artifact → ToolMessage.artifact → shudhu tomar code dekhe (model dekhe na)
// Eta tomar CRM-e perfect fit: model shudhu summary dekhe, kintu frontend
// er kache full row/id gula pathate paro token pora charai.
export const listInvoices = tool(
  async ({ clientName }) => {
    const rows = [
      { id: "INV-1041", amount: 2400, status: "paid" },
      { id: "INV-1042", amount: 980, status: "overdue" },
      { id: "INV-1043", amount: 1500, status: "draft" },
    ];
    const overdue = rows.filter((r) => r.status === "overdue");

    return [
      // model er jonno — choto, summarised
      `${clientName}: ${rows.length} invoices, ${overdue.length} overdue ` +
        `(${overdue.map((r) => r.id).join(", ") || "none"}).`,
      // app er jonno — full payload
      { clientName, rows, generatedFor: "ui-table" },
    ];
  },
  {
    name: "list_invoices",
    description: "List invoices for a client.",
    schema: z.object({ clientName: z.string() }),
    responseFormat: "content_and_artifact",
  },
);

// ── [T6] returnDirect: true ───────────────────────────────────────────────────
// Tool result-i final answer — model ke ar ekbar call kora hoy na. Ekta model
// call bachay. Caution: parallel tool call hole SHOB tool returnDirect hole-i
// loop ta break kore.
export const escalateToHuman = tool(
  async ({ reason }) =>
    `Escalated to a human agent. Reason: ${reason}. Someone will reply within 2 hours.`,
  {
    name: "escalate_to_human",
    description:
      "Hand the conversation to a human when you cannot resolve it. " +
      "This ends your turn immediately.",
    schema: z.object({ reason: z.string() }),
    returnDirect: true,
  },
);

// ── [T6] ToolRuntime — 2nd argument ───────────────────────────────────────────
// runtime.context    → per-run data (createAgent-er contextSchema theke)
// runtime.writer     → custom stream event (streamMode "custom" lage)
// runtime.toolCallId → ei call er id
// runtime.state      → agent state
// runtime.store      → long-term memory (LangGraph, ekhane use korchi na)
export const bookMeeting = tool(
  async ({ title, durationMinutes }, runtime) => {
    // Progress event — UI te "step 1/2 chalche" dekhate.
    runtime.writer?.({ type: "progress", tool: "book_meeting", step: "checking calendar" });

    const { userName, timezone } = runtime.context ?? {};

    runtime.writer?.({ type: "progress", tool: "book_meeting", step: "creating event" });

    return (
      `Booked "${title}" (${durationMinutes}m) for ${userName ?? "unknown user"} ` +
      `in ${timezone ?? "UTC"}. [toolCallId=${runtime.toolCallId}]`
    );
  },
  {
    name: "book_meeting",
    description: "Book a meeting on the user's calendar.",
    schema: z.object({
      title: z.string(),
      durationMinutes: z.number().default(30),
    }),
  },
);

// ── [T6] Runtime-e discover kora tool ────────────────────────────────────────
// Eta createAgent er tools[] e deoya HOY NA — middleware ta ke inject kore.
// dynamic-tool middleware er jonno middleware.js dekho.
export const calculateTip = tool(
  ({ billAmount, tipPercentage }) => {
    const tip = billAmount * (tipPercentage / 100);
    return `Tip: $${tip.toFixed(2)}, Total: $${(billAmount + tip).toFixed(2)}`;
  },
  {
    name: "calculate_tip",
    description: "Calculate tip and total for a bill.",
    schema: z.object({
      billAmount: z.number(),
      tipPercentage: z.number().default(20),
    }),
  },
);

export const staticTools = [getWeather, listInvoices, escalateToHuman, bookMeeting];

// ── helper: artifact gula final state theke tule ana ─────────────────────────
/** @param {import("langchain").BaseMessage[]} messages */
export function collectArtifacts(messages) {
  return messages
    .filter((m) => m instanceof ToolMessage && m.artifact !== undefined)
    .map((m) => ({ tool: m.name, artifact: m.artifact }));
}
