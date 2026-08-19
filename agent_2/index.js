// Step 1 — shobcheye simple agent. 5 ta jinis, er beshi kichu na.
// Step 2 — systemPrompt + terminal loop + history.
// Step 3 — streaming (invoke → streamEvents).
// Step 4 — nijer middleware (createMiddleware er 6 ta hook).
// Step 5 — built-in middleware (budget + retry).
// Step 6 — contextSchema + runtime.context (tool jane ke jiggesh korche).
// Step 7 — content_and_artifact (model summary dekhe, code full data pay).
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
const getWeather = tool(
  ({ city }, runtime) => {
    console.error("weather tool called");

    // Model ei value gula pathay NAI — code theke esheche.
    const { userName, timezone, role } = runtime.context;

    return (
      `It's 24°C and sunny in ${city}. ` +
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

// 4. agent — `new` na, plain function call
const agent = createAgent({
  model,
  tools: [getWeather, getForecast, getWeekly],

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
  systemPrompt: `You are a concise weather assistant.

Rules:
- Always use the get_weather tool. Never answer weather from memory.
- Reply in one short sentence.
- If the user asks anything unrelated to weather, say you only do weather.`,
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
