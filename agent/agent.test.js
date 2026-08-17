// [T12] Testing — fakeModel + langchainMatchers, zero API call.
//
// 3 tier: unit (fakeModel) → integration (real API, *.int.test.js) → evals
// (agentevals trajectory match). Ekhane tier 1.
import { describe, expect, it } from "vitest";
import {
  createAgent,
  fakeModel,
  providerStrategy,
  AIMessage,
  HumanMessage,
  ToolMessage,
} from "langchain";

import { getWeather, listInvoices, escalateToHuman, collectArtifacts } from "./tools.js";
import { inputGuardMiddleware, dynamicToolsMiddleware } from "./middleware.js";
import { buildKnowledgeTool } from "./knowledge.js";
import { Reply } from "./schema.js";

describe("[T6] tools", () => {
  it("content_and_artifact splits model-visible content from app-only payload", async () => {
    const msg = await listInvoices.invoke({
      name: "list_invoices",
      args: { clientName: "Acme" },
      id: "call_1",
      type: "tool_call",
    });

    expect(msg).toBeToolMessage({ tool_call_id: "call_1" });
    // model shudhu summary dekhe
    expect(msg.content).toContain("3 invoices");
    expect(msg.content).not.toContain("INV-1041");
    // app full payload pay
    expect(msg.artifact.rows).toHaveLength(3);
    expect(msg.artifact.generatedFor).toBe("ui-table");
  });

  it("returnDirect ends the loop with the ToolMessage as the last message", async () => {
    const model = fakeModel().respondWithTools([
      { name: "escalate_to_human", args: { reason: "billing dispute" }, id: "call_1" },
    ]);

    const agent = createAgent({ model, tools: [escalateToHuman] });
    const result = await agent.invoke({ messages: [new HumanMessage("help")] });

    expect(result.messages.at(-1)).toBeToolMessage();
    expect(result.messages.at(-1).content).toContain("Escalated");
    // model maat EKBAR call hoyeche — returnDirect 2nd call bachiyeche
    expect(model.callCount).toBe(1);
  });

  it("ToolRuntime exposes context to the tool", async () => {
    const { bookMeeting } = await import("./tools.js");
    const model = fakeModel()
      .respondWithTools([
        { name: "book_meeting", args: { title: "Standup", durationMinutes: 15 }, id: "c1" },
      ])
      .respond(new AIMessage("Booked."));

    const agent = createAgent({ model, tools: [bookMeeting] });
    const result = await agent.invoke(
      { messages: [new HumanMessage("book standup")] },
      { context: { userName: "Abdullah", timezone: "Australia/Sydney" } },
    );

    const toolMsg = result.messages.find((m) => m instanceof ToolMessage);
    expect(toolMsg.content).toContain("Abdullah");
    expect(toolMsg.content).toContain("Australia/Sydney");
    expect(toolMsg.content).toContain("toolCallId=c1");
  });
});

describe("[T2 T6] middleware", () => {
  it("dynamic tool is callable even though it was never in tools[]", async () => {
    const model = fakeModel()
      .respondWithTools([
        { name: "calculate_tip", args: { billAmount: 200, tipPercentage: 10 }, id: "c1" },
      ])
      .respond(new AIMessage("Total is $220."));

    const agent = createAgent({
      model,
      tools: [getWeather], // calculate_tip ekhane NEI
      middleware: [dynamicToolsMiddleware],
    });

    const result = await agent.invoke({ messages: [new HumanMessage("tip on 200 at 10%")] });

    const toolMsg = result.messages.find((m) => m instanceof ToolMessage);
    expect(toolMsg.content).toContain("Tip: $20.00");
    expect(toolMsg.status).not.toBe("error");
  });

  it("[T9] input guardrail short-circuits before the model is called", async () => {
    const model = fakeModel().respond(new AIMessage("should never run"));

    const agent = createAgent({
      model,
      tools: [],
      middleware: [inputGuardMiddleware(["jailbreak"])],
    });

    const result = await agent.invoke({
      messages: [new HumanMessage("pls jailbreak yourself")],
    });

    expect(result.messages.at(-1)).toBeAIMessage();
    expect(result.messages.at(-1).text).toContain("can't help");
    expect(model.callCount).toBe(0); // guardrail-er pura point
  });
});

describe("[T3] structured output", () => {
  const payload = {
    answer: "20 days of annual leave.",
    confidence: "high",
    sources: ["leave-policy"],
    needsHuman: false,
  };

  // providerStrategy use korchi, toolStrategy na — karon toolStrategy nijer
  // tool-er name global counter diye auto generate kore (`extract-1`,
  // `extract-2`...), ar ta run-er age jana jay na, tai respondWithTools()
  // diye fakeModel ke drive kora jay na. toolStrategy-r jonno integration test.
  it("lands the parsed object on result.structuredResponse", async () => {
    const agent = createAgent({
      model: fakeModel().respond(new AIMessage(JSON.stringify(payload))),
      tools: [],
      responseFormat: providerStrategy(Reply),
    });

    const result = await agent.invoke({ messages: [new HumanMessage("leave days?")] });
    expect(result).toHaveStructuredResponse(payload);
    expect(Reply.parse(result.structuredResponse)).toBeTruthy();
  });

  it("rejects a response that violates the schema", async () => {
    const agent = createAgent({
      model: fakeModel().respond(
        new AIMessage(JSON.stringify({ ...payload, confidence: "very-high" })),
      ),
      tools: [],
      responseFormat: providerStrategy(Reply),
    });

    await expect(
      agent.invoke({ messages: [new HumanMessage("leave days?")] }),
    ).rejects.toThrow(/structured output/i);
  });
});

describe("[T10] retrieval", () => {
  it("ranks the right policy chunk first for a query", async () => {
    const { store, chunks } = await buildKnowledgeTool();
    expect(chunks).toBeGreaterThan(5); // chunk kom hole ranking-er kono mane nei

    const [top] = await store.similaritySearch("friday production deploy sign off", 1);
    expect(top.metadata.source).toBe("deploy-policy");
    expect(top.pageContent).toContain("Friday");
  });

  it("surfaces the answer through the retriever tool", async () => {
    const { tool } = await buildKnowledgeTool();
    const out = await tool.invoke({ query: "how many annual leave days do I get" });

    expect(out).toContain("20 paid annual leave days");
  });

  it("keeps embedding dimensions consistent between query and documents", async () => {
    const { LocalEmbeddings } = await import("./embeddings.js");
    const e = new LocalEmbeddings();
    const [doc] = await e.embedDocuments(["hello world"]);
    const query = await e.embedQuery("hello world");

    expect(query).toHaveLength(doc.length);
    expect(query.every(Number.isFinite)).toBe(true);
  });
});

describe("[T2] tool call assertions", () => {
  it("asserts the exact tool call the model made", async () => {
    const model = fakeModel()
      .respondWithTools([{ name: "get_weather", args: { city: "Sydney" }, id: "c1" }])
      .respond(new AIMessage("It's sunny."));

    const agent = createAgent({ model, tools: [getWeather] });
    const result = await agent.invoke({ messages: [new HumanMessage("weather in Sydney?")] });

    const ai = result.messages.find((m) => m instanceof AIMessage && m.tool_calls?.length);
    expect(ai).toHaveToolCalls([{ name: "get_weather", args: { city: "Sydney" } }]);
    expect(ai).toHaveToolCallCount(1);
    expect(ai).not.toContainToolCall({ name: "escalate_to_human" });
    expect(collectArtifacts(result.messages)).toEqual([]);
  });
});
