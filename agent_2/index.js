// Step 1 — shobcheye simple agent. 5 ta jinis, er beshi kichu na.
// Step 2 — systemPrompt + terminal loop + history.
// Step 3 — streaming (invoke → streamEvents).
// Step 4 — nijer middleware (createMiddleware er 6 ta hook).
// Step 5 — built-in middleware (budget + retry).
// Step 6 — contextSchema + runtime.context (tool jane ke jiggesh korche).
// Step 7 — content_and_artifact (model summary dekhe, code full data pay).
// Step 8 — structured output (uttor string na, typed object).
// Step 9 — RAG (splitter → embeddings → vector store → retriever tool).
// Step 10 — subagent (ekta agent ke tool baniye onno agent-e deoya).
import { ChatOpenRouter } from "@langchain/openrouter";
import { config } from "dotenv";
import {
  createAgent,
  createMiddleware,
  modelCallLimitMiddleware,
  tool,
  toolCallLimitMiddleware,
  toolRetryMiddleware,
} from "langchain";
// [Step 9] RAG — 3 ta ALADA package. v1-e `langchain/vectorstores/memory`
// NEI, legacy shob kichu @langchain/classic e chole geche.
import { Embeddings } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import readline from "readline/promises";
import * as z from "zod";

// 1. .env load
// NOTE: config() `.env` ta CWD theke khoje, ei file-er pashe theke na.
// Tai project folder theke chalate hobe. Kothao theke chalate chaile:
//   config({ path: new URL("../.env", import.meta.url) })
config();

// 2. model
const model = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  model: "deepseek/deepseek-v4-pro",
});

// 3. tool
// [Step 6] 2nd argument `runtime` — ekhane-i context, writer, toolCallId.
// [Step 8] Tool ta ekhon HONEST — ja jane na, ta bole dey. Age eta jekono
// city-r jonno "24°C and sunny" ferot dito, tai model "Mars"-eo confident
// uttor dito. Tool jodi mithye na bole, tobei schema-r `confidence` field-er
// kono mane thake.
const KNOWN = { Dhaka: 24, Sylhet: 26, Chattogram: 28, Sydney: 19 };

const getWeather = tool(
  ({ city }, runtime) => {
    console.error("weather tool called");

    // Model ei value gula pathay NAI — code theke esheche.
    const { userName, timezone, role } = runtime.context;

    const temp = KNOWN[city];
    if (temp === undefined) {
      return `No weather station for "${city}". Known: ${Object.keys(KNOWN).join(", ")}.`;
    }

    return (
      `It's ${temp}°C and sunny in ${city}. ` +
      `(for ${userName}, tz ${timezone}, role ${role})`
    );
  },
  {
    name: "get_weather",
    description: "Get the current weather in a given location",
    schema: z.object({
      city: z.string().describe("The city to get the weather for"),
    }),
  },
);

// ─── [Step 6] contextSchema ─────────────────────────────────────────────────
//
// Per-run data — ke request korche, tar role, timezone. Prottek request e alada.
//
// KENO tool-er schema te ei gula rakhi na:
//   1. Model tokhon `userId` nijer theke banabe ba bhul ta dibe
//   2. Model ke user "ami admin" bole convince kore felte parbe
//   3. Sudhu sudhu token — model-er ei data janar kono dorkar nei
//
// contextSchema diye pathale model eta DEKHE-I NA. Tumi code theke dao,
// tool code theke pore. Model-er hat diye jay na.
//
// Plain zod use korchi — LangGraph-er StateSchema na, tai eta LangGraph-free.
const contextSchema = z.object({
  userName: z.string(),
  role: z.enum(["staff", "admin"]),
  timezone: z.string().default("Asia/Dhaka"),
});

// [Step 5] Ekta iccha kore flaky tool — noile retry middleware kaj korche
// kina dekha jabe na. Prothom 2 bar throw kore, 3rd bar e kaj kore.
let forecastAttempts = 0;
const getForecast = tool(
  ({ city }) => {
    forecastAttempts += 1;
    if (forecastAttempts < 3) {
      throw new Error(`upstream forecast API timed out (attempt ${forecastAttempts})`);
    }
    return `${city}: tomorrow 26°C, light rain in the evening.`;
  },
  {
    name: "get_forecast",
    description: "Get tomorrow's weather forecast for a city.",
    schema: z.object({ city: z.string() }),
  },
);

// ─── [Step 7] content_and_artifact ──────────────────────────────────────────
//
// Ei tool ta 7 din-er data ferot dey. Puro data model ke dile ~400 token,
// ar model tar bhitor theke asol kaj hariye fele. Tai duita jinis:
//
//   return [ content, artifact ]
//            ↑         ↑
//            │         └─ tomar CODE dekhe. model NA. 0 token.
//            └─ model dekhe. choto summary.
//
// `responseFormat: "content_and_artifact"` na dile ei array ta-i stringify
// hoye model er kache chole jabe.
const getWeekly = tool(
  ({ city }) => {
    // Bhaan korchi eta DB / API theke ashche.
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d, i) => ({
      day: d,
      high: 26 + (i % 4),
      low: 18 + (i % 3),
      rainChance: (i * 13) % 100,
      icon: i % 2 ? "cloud" : "sun",
    }));

    const wettest = days.reduce((a, b) => (b.rainChance > a.rainChance ? b : a));

    return [
      // [0] content — model ei tuku dekhe. ~25 token.
      `${city} 7-day: highs ${Math.min(...days.map((d) => d.high))}-${Math.max(
        ...days.map((d) => d.high),
      )}°C. Wettest day ${wettest.day} (${wettest.rainChance}% rain).`,

      // [1] artifact — model kokhono dekhe na. Chart/table render korar
      //     jonno tomar frontend ei ta pabe.
      { city, days, generatedFor: "weekly-chart" },
    ];
  },
  {
    name: "get_weekly",
    description: "Get the 7-day weather outlook for a city.",
    schema: z.object({ city: z.string() }),
    responseFormat: "content_and_artifact",
  },
);

// ─── [Step 9] RAG ───────────────────────────────────────────────────────────
//
// RAG = 4 ta step, ar ekta-o magic na:
//   1. Boro document ke CHOTO chunk e bhango       (splitter)
//   2. Proti chunk ke ekta number array e badlao   (embeddings)
//   3. Number array gula ekta store e rakho        (vector store)
//   4. Prosno-o number array kore, kachakachi khojo (retriever)
//
// ── 2 nombor niye ekta problem ──────────────────────────────────────────────
// OpenRouter er embeddings endpoint NEI. Tomar .env e shudhu OPENROUTER_API_KEY.
// Tai nijer ekta Embeddings class likhchi — offline, key lage na.
//
// Embeddings base class e maat 2 ta method implement korte hoy.
// Production e ei class ta bodle OpenAIEmbeddings / OllamaEmbeddings / Voyage
// boshalei cholbe, baki code ek-i thakbe.
const DIM = 256;

function embedOne(text) {
  const vec = new Float64Array(DIM);

  // Text ke word e bhango, proti word ke hash kore ekta slot e felo.
  for (const token of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let h = 0x811c9dc5; // FNV-1a hash
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    vec[h % DIM] += 1;
  }

  // L2 normalise — noile lomba document choto-r cheye beshi score pabe.
  let norm = 0;
  for (let i = 0; i < DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;

  return Array.from(vec, (v) => v / norm);
}

class LocalEmbeddings extends Embeddings {
  constructor(fields = {}) {
    super(fields); // lagbei — this.caller (retry/concurrency) set kore
  }
  async embedDocuments(texts) {
    return texts.map(embedOne);
  }
  async embedQuery(text) {
    // ⚠️ embedQuery ar embedDocuments-er DIMENSION ek hote hobe. Store
    // validate kore na — mismatch hole chupchap bhul result debe.
    return embedOne(text);
  }
}

// NOTE: eta LEXICAL matching (shobdo mile), semantic na. "holiday" likhle
// "leave policy" khuje pabe na. Real embeddings model oita parbe. Learning
// er jonno ei tuku jothesto — pipeline ta ek-i.

const HANDBOOK = [
  {
    source: "umbrella-policy",
    text: `Umbrella policy. Staff may claim one umbrella per calendar year, up to 900 BDT.
Claims require a receipt uploaded to the HR portal within 30 days of purchase.
Umbrellas lost on company premises are replaced free of charge, once per year.
Golf umbrellas are not covered because they do not fit in the office lockers.`,
  },
  {
    source: "rain-day-policy",
    text: `Rain day policy. When rainfall exceeds 50mm before 7am, the office opens at 11am.
Staff who cannot reach the office on a declared rain day may work from home without
using a leave day. The rain day declaration is sent by SMS before 7:30am.
Client meetings on a rain day must be moved to video call, not cancelled.`,
  },
  {
    source: "heat-policy",
    text: `Heat policy. When the forecast high exceeds 38 degrees Celsius, outdoor site
visits are suspended for that day. Staff on site must return to the office by noon.
Air-conditioned transport is reimbursed in full on declared heat days.
Heat days do not count against the annual leave allowance.`,
  },
];

async function buildHandbookTool() {
  // 1. Split. chunkSize CHOTO rakhchi iccha kore — boro rakhle 3 ta policy
  //    = 3 ta chunk, ar k:3 mane retriever SHOB kichu ferot dey. Tokhon
  //    ranking-er kono mane thake na, RAG-o kaj korche kina bujha jay na.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 220,
    chunkOverlap: 40,
  });

  const splits = await splitter.splitDocuments(
    HANDBOOK.map(
      (d) => new Document({ pageContent: d.text, metadata: { source: d.source } }),
    ),
  );

  // 2 + 3. Embed kore store e rakho. Ei line-tai embedDocuments() dake.
  const store = new MemoryVectorStore(new LocalEmbeddings());
  await store.addDocuments(splits);

  // 4. Retriever ke tool banao. `k: 3` = shobcheye kachakachi 3 ta chunk.
  //    ⚠️ createRetrieverTool er schema FIXED — shudhu { query: string }.
  //    Nijer schema (filter, date range) chaile tool() diye hate likhte hobe
  //    ar bhitore retriever.invoke() call korte hobe.
  const searchHandbook = createRetrieverTool(store.asRetriever({ k: 3 }), {
    name: "search_handbook",
    description:
      "Search the internal company handbook for weather-related policies " +
      "(umbrella claims, rain days, heat days). Use this for any question " +
      "about company rules instead of answering from memory.",
  });

  return { searchHandbook, store, chunks: splits.length };
}

// Top-level await — app start e EKBAR index hoy, proti prosno te na.
const { searchHandbook, chunks } = await buildHandbookTool();
console.error(`[rag] handbook indexed: ${chunks} chunks`);

// ─── [Step 10] subagent ─────────────────────────────────────────────────────
//
// Subagent kono notun concept NA. Eta shudhu:
//   ekta createAgent → tool() diye mora → onno agent-er tools[] e deoya
//
// Mane parent agent-er kache eta ekta SADHARON TOOL. Se jane na bhitore
// arekta LLM chalche.
//
// KENO lagbe — 3 ta karon:
//
//   1. TOOL BHIR. Parent er kache 4 ta tool. 40 ta hole model gulie fele.
//      10 ta subagent, protita 4 ta tool niye → parent dekhe 1 ta tool.
//
//   2. CONTEXT PORISKAR. Subagent nijer alada message list e kaj kore.
//      Se 10 bar handbook search korle oi 10 ta tool result parent-er
//      history te DHOKE NA — shudhu final summary ta dhoke.
//
//   3. ALADA PROMPT / MODEL. Subagent ke cheap model + kora prompt dite paro.
//
// ⚠️ Ja ekhane use korchi NA (egula LangGraph, alada repo-r jonno):
//   - getCurrentTaskInput() diye parent-er state pora → bodole arg pathacchi
//   - new Command({ update }) diye parent-er state lekha → bodole string return
//   - subagent-er nijer checkpointer / interrupt

// Subagent 1 — handbook ghete report dey. Parent-er cheye kora prompt.
const researcher = createAgent({
  name: "researcher",
  model,
  tools: [searchHandbook],
  systemPrompt: `You are a policy research subagent.

- Search the handbook, possibly several times with different wordings.
- Report ONLY what the handbook actually says, with the policy name.
- If the handbook does not cover it, say "not covered" plainly. Never guess.
- Output bullet points. No preamble.`,
});

// Subagent 2 — rough note ke poriskar bakko banay. Tool lage na.
const writer = createAgent({
  name: "writer",
  model,
  tools: [],
  systemPrompt: `You are a writing subagent.
Rewrite the given notes as a clear, short reply for a staff member.
Plain language. No marketing tone. Never invent facts that are not in the notes.`,
});

const SUBAGENTS = { researcher, writer };

// EKTA dispatch tool, proti agent-er jonno alada tool na. Keno: 10 ta agent
// hole 10 ta tool = abar shei tool bhir. Ekta tool + ekta enum = 1 ta slot.
const task = tool(
  async ({ agentName, description }) => {
    const sub = SUBAGENTS[agentName];
    if (!sub) return `Unknown agent "${agentName}". Available: ${Object.keys(SUBAGENTS).join(", ")}.`;

    // Notun message list — parent-er history EKHANE ashe na.
    const r = await sub.invoke({ messages: [{ role: "user", content: description }] });

    // Shudhu final text ta parent ke ferot jay. Bhitorer tool call gula na.
    return r.messages.at(-1)?.text ?? "(subagent returned nothing)";
  },
  {
    name: "task",
    description: `Delegate a self-contained task to a subagent.

Available agents:
- researcher: search the company handbook and report findings
- writer: turn rough notes into a clear short reply

IMPORTANT: the subagent CANNOT see this conversation. It only sees your
description. Put every fact it needs into the description.`,
    schema: z.object({
      agentName: z.enum(["researcher", "writer"]),
      description: z.string().describe("Self-contained task description"),
    }),
  },
);

// ─── [Step 8] structured output ─────────────────────────────────────────────
//
// `.describe()` gula shudhu comment na — model EI GULA PORE. Schema ta-i
// tomar prompt. Bhalo describe = bhalo output.
const WeatherReply = z.object({
  answer: z.string().describe("One short sentence to show the user"),
  city: z.string().describe("The city the answer is about"),
  tempC: z.number().nullable().describe("Temperature in Celsius, null if unknown"),
  confidence: z
    .enum(["high", "medium", "low"])
    .describe("low if you could not use a tool and had to guess"),
  needsFollowUp: z.boolean().describe("true if the question was ambiguous"),
});

// ─── [Step 4] nijer middleware ──────────────────────────────────────────────
//
// Middleware = agent-er loop-er majhe tomar code dhukanor jayga.
// 6 ta hook, ar egula 2 rokom:
//
//   NODE hook   — beforeAgent / beforeModel / afterModel / afterAgent
//                 Kichu ghotar AGE ba PORE chole. State bodlate paro,
//                 ba `jumpTo` diye loop thamiye dite paro.
//
//   WRAP hook   — wrapModelCall / wrapToolCall
//                 Ghotona take GHIRE chole. Tumi `handler()` call koro.
//                 Call na korle ghotona ta ghotbei na. Try/catch, timing,
//                 retry, request modify — shob ekhane.
//
// Order:  before* → array-r prothom theke shesh
//         after*  → ULTA, shesh theke prothom
//         wrap*   → nested, prothom middleware baki shob ke ghire rakhe
//
// stderr e log korchi, stdout e na — noile streaming token er sathe mishe jabe.
// `2>/dev/null` diye off korte parba.
const log = (...a) => console.error("  \x1b[2m│", ...a, "\x1b[0m");

let modelCalls = 0;

const traceMiddleware = createMiddleware({
  name: "TraceMiddleware",

  // 1. Puro run e EKBAR, shob kichur age.
  beforeAgent: (state) => {
    modelCalls = 0;
    log(`beforeAgent · history te ${state.messages.length} ta message`);
  },

  // 2. PROTTEK model call er age.
  beforeModel: (state) => {
    log(`beforeModel #${modelCalls + 1} · ${state.messages.length} ta message pathacchi`);
  },

  // 3. Model call ke GHIRE. request ekhane bodlano jay.
  wrapModelCall: async (request, handler) => {
    log(`wrapModelCall · model ke ${request.tools.length} ta tool dekhacchi`);

    const t0 = Date.now();
    const response = await handler(request); // ← ei line-i asol model call
    log(`wrapModelCall · ${Date.now() - t0}ms · toolCalls=${response.tool_calls?.length ?? 0}`);

    return response;
  },

  // 4. Prottek model response er pore.
  afterModel: () => {
    modelCalls += 1;
  },

  // 5. Prottek tool call ke GHIRE. Timing, audit log, error handling.
  wrapToolCall: async (request, handler) => {
    const t0 = Date.now();
    log(`tool → ${request.toolCall.name}(${JSON.stringify(request.toolCall.args)})`);
    try {
      const result = await handler(request);
      log(`tool ← ok · ${Date.now() - t0}ms`);
      return result;
    } catch (err) {
      log(`tool ✗ ${err.message}`);
      throw err;
    }
  },

  // 6. Puro run e EKBAR, shesh e.
  afterAgent: () => {
    log(`afterAgent · moot ${modelCalls} ta model call`);
  },
});

// 4. agent config. [Step 8] duita agent lagbe, tai config ta alada kore
// rakhchi ar duijaygay spread korbo.
const shared = {
  model,
  tools: [getWeather, getForecast, getWeekly, searchHandbook, task],

  // [Step 6] declare kore dite hobe, noile runtime.context faka thakbe.
  contextSchema,

  // [Step 4/5] middleware ekta ARRAY — order matter kore.
  //
  // before* → upor theke niche.   after* → niche theke upor (ulta).
  // Tai critical jinis (budget, guardrail) SHURUTE rakho.
  middleware: [
    // 1. Budget — runaway loop thamay. Tomar CRM-er hate-lekha step-budget
    //    er official version. `exitBehavior: "end"` mane limit-e pouchale
    //    crash na kore gracefully thame.
    //    NOTE: `threadLimit` option-o ache, kintu tar jonno checkpointer
    //    (= LangGraph) lage. `runLimit` ek run e koto — eta LangGraph-free.
    modelCallLimitMiddleware({ runLimit: 6, exitBehavior: "end" }),
    toolCallLimitMiddleware({ runLimit: 8, exitBehavior: "continue" }),

    // 2. Retry — tool throw korle abar chesta kore, exponential backoff diye.
    //    `onFailure: "continue"` mane shob retry fail korleo crash na —
    //    error ta ToolMessage hoye model er kache jay, model decide kore.
    toolRetryMiddleware({
      maxRetries: 3,
      initialDelayMs: 300,
      onFailure: "continue",
    }),

    // 3. Trace shesh e — er upore ja ache shob ke ghire rakhbe... na.
    //    Ulta: array-r SHESHE thakle eta shobar BHITORE thakbe.
    traceMiddleware,
  ],

  // [Step 2] systemPrompt — model ke ki bhabe behave korbe bola.
  // Prottek model call e ei ta pathano hoy, tomake bar bar dite hobe na.
  systemPrompt: `You are a concise weather assistant for company staff.

Rules:
- For current weather, use get_weather. Never answer weather from memory.
- For questions about COMPANY POLICY (umbrella claims, rain days, heat days),
  you MUST call search_handbook first. Never answer policy from memory.
- If the handbook does not cover it, say so plainly instead of guessing.
- For a question that needs SEVERAL policies compared, or a long answer that
  must be rewritten cleanly, use the task tool to delegate to a subagent.
- Reply in one or two short sentences.
- If the user asks anything unrelated to weather or weather policy, say so.`,
};

// Normal agent — streaming er jonno. responseFormat NEI.
const agent = createAgent({ ...shared, name: "weather" });

// [Step 8] Structured agent — `responseFormat` deoya.
//
// KENO alada instance: responseFormat set thakle model-er shesh output ta
// ekta JSON tool call hoy, text na. Mane `message.text` FAKA thakbe ar
// streaming er kono mane thake na. Tai ei ta `invoke()` diye chalabo.
//
// Bare zod schema pathacchi — LangChain nijei decide kore:
//   model native JSON mode support kore → providerStrategy
//   kore na                            → toolStrategy e fallback
// Force korte chaile `toolStrategy(WeatherReply)` ba `providerStrategy(...)`.
const structuredAgent = createAgent({
  ...shared,
  name: "weather-structured",
  responseFormat: WeatherReply,
});

// ─── [Step 2] terminal loop + history ───────────────────────────────────────

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Conversation memory. Agent nijer theke kichu mone rakhe na — prottek bar
// puro history take dite hobe.
let history = [];

// ─── [Step 3] streaming ─────────────────────────────────────────────────────
//
// invoke() shesh na howa porjonto kichu dey na. streamEvents() ekta object dey
// jekhane alada alada TYPED iterator ache:
//   stream.messages   → proti LLM call; .text / .reasoning / .toolCalls / .usage
//   stream.toolCalls  → tool execution; .name .input, await .output
//   stream.output     → final state (invoke() ja dito, ei tai)
// [Step 6] Per-run config. Real app e eta session / JWT theke ashbe.
// `.parse()` kore nicchi — bhul shape hole EKHANE throw korbe, tool-er
// bhitore giye `undefined` hoye na.
const runConfig = {
  context: contextSchema.parse({
    userName: "Abdullah",
    role: "admin",
    timezone: "Asia/Dhaka",
  }),
};

async function streamRun(input) {
  const stream = await agent.streamEvents(input, { ...runConfig, version: "v3" });

  // ⚠️ Promise.all LAGBEI. Duita `for await` ek er por ek likhle dwitiyo ta
  // starve kore — prothom iterator ta shob event kheye fele. Egula
  // CONCURRENTLY consume korte hobe.
  await Promise.all([
    (async () => {
      for await (const message of stream.messages) {
        for await (const token of message.text) {
          process.stdout.write(token);
        }
      }
    })(),

    (async () => {
      for await (const call of stream.toolCalls) {
        process.stdout.write(`\n[tool] ${call.name} ${JSON.stringify(call.input)}\n`);

        // ⚠️ `call.output` ekta PROMISE, ar tool throw korle eta REJECT kore.
        // try/catch na dile unhandled rejection hoy ar puro process mara jay —
        // jodio agent nijei error ta handle korte parto (retry / model ke
        // ferot pathano). Ei bug ta hate hate khelam.
        try {
          process.stdout.write(`[tool] ↳ ${await call.output}\n`);
        } catch (err) {
          process.stdout.write(`[tool] ✗ ${err.message}\n`);
        }
      }
    })(),
  ]);

  // Stream sesh — final state. invoke() er return value-r shathe ek.
  return await stream.output;
}

// [Step 7] Final message list theke artifact gula tule ana. Plain JS —
// kono LangChain API na. Ei ta-i tomar frontend e pathabe.
function collectArtifacts(messages) {
  return messages
    .filter((m) => m.getType() === "tool" && m.artifact !== undefined)
    .map((m) => ({ tool: m.name, artifact: m.artifact }));
}

console.log("Weather agent. `q` diye ber hobe.\n");

while (true) {
  const question = await rl.question("› ");
  if (question.trim() === "q") break;
  if (!question.trim()) continue;
  console.log("\n\n\n", { history }, "\n\n\n");

  // [Step 8] `/json <prosno>` — structured output. Streaming nei, invoke().
  if (question.startsWith("/json ")) {
    const r = await structuredAgent.invoke(
      { messages: [...history, { role: "user", content: question.slice(6) }] },
      runConfig,
    );
    history = r.messages;

    // Parsed object ekhane. responseFormat na dile ei key ta thake-i na.
    console.log(JSON.stringify(r.structuredResponse, null, 2), "\n");
    continue;
  }

  const result = await streamRun({
    messages: [...history, { role: "user", content: question }],
  });

  // ⚠️ push na, REPLACE. Agent nijei tool call + tool result + AI reply
  // shob `result.messages` e jure diyeche — push korle duplicate hobe.
  history = result.messages;

  // uttor ta already stream hoye geche, tai ar print korchi na — shudhu
  // ekta notun line.
  console.log("\n");

  // [Step 7] Artifact — real app e eta HTTP response e frontend ke pathabe,
  // console e chapbe na.
  const artifacts = collectArtifacts(result.messages);
  if (artifacts.length) {
    console.log("[artifacts]", JSON.stringify(artifacts, null, 2), "\n");
  }
}

rl.close();
