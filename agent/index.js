// Entry point — REPL je 13 ta topic exercise kore.
//
// [T7] Messages — HumanMessage/contentBlocks/multimodal ar .text/.contentBlocks
// diye safely pora ekhane ache.
//
// Conversation memory INTENTIONALLY hate maintain korchi (history array),
// checkpointer diye na — karon checkpointer/MemorySaver LangGraph, ar seta
// tomar alada repo-r jonno rakha. Ekhane dekhbe checkpointer thakle ki ki
// boilerplate ta se sarabe.
import { config } from "dotenv";
import readline from "readline/promises";
import { HumanMessage } from "langchain";

import { buildAgents, contextSchema } from "./build.js";
import { streamRun, streamCustomEvents } from "./stream.js";
import { collectArtifacts } from "./tools.js";
import { formatUsage } from "./model.js";
import { closeMcp } from "./mcp.js";

config();

if (!process.env.OPENROUTER_API_KEY) {
  console.error("OPENROUTER_API_KEY missing in .env");
  process.exit(1);
}

const bold = (s) => `\x1b[1m${s}\x1b[0m`;
const dim = (s) => `\x1b[2m${s}\x1b[0m`;
const green = (s) => `\x1b[32m${s}\x1b[0m`;

const HELP = `
${bold("Commands")}
  <question>              stream a normal answer            [T1 T2 T4 T6 T10 T13]
  /json <question>        structured output                 [T3]
  /custom <question>      show runtime.writer progress events   [T4 T6]
  /image <url> | <q>      multimodal message                [T7]
  /artifacts              last turn's tool artifacts        [T6]
  /blocks                 last reply's content blocks       [T7]
  /reset                  clear conversation history
  /help                   this
  q                       quit
`;

const agents = await buildAgents({ verbose: process.env.QUIET !== "1" });

console.log(bold("\nLangChain v1 demo agent"));
console.log(dim(`tools: ${agents.tools.map((t) => t.name).join(", ")}`));
console.log(dim(`middleware: ${agents.middlewareCount} · ${agents.notes.join(" · ")}`));
console.log(HELP);

// [T5] Per-run context — tool ar middleware `runtime.context` diye pore.
const runConfig = {
  context: contextSchema.parse({
    userName: "Abdullah",
    role: "admin",
    timezone: "Australia/Sydney",
  }),

  // ⚠️ EI TA NA DILE agent crash kore — hate hate debug kore ber korlam.
  //
  // recursionLimit agent LOOP iteration count na, GRAPH STEP count. Ar
  // prottek middleware hook (beforeModel/afterModel/...) ekta alada graph
  // node. Tai 11 ta middleware diye 1 ta model turn ~12+ step khay.
  //
  // Measured: 11 middleware + 4 model call → 50 step-e fail, 100-e pass.
  // Default 25 e maat 2 ta model turn-eo kulay na.
  //
  // Rule of thumb: recursionLimit ≳ (expected model turns) × (middleware
  // count + 3). Semantic budget-er jonno modelCallLimitMiddleware use koro
  // (oita gracefully "end" kore); recursionLimit shudhu mechanical backstop,
  // ar limit chharale eta GraphRecursionError THROW kore.
  recursionLimit: 150,
};

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

// Interactive (TTY) hole rl.question() diye prompt kori. Piped hole
// (`printf '...' | node agent/index.js`) rl.question kaj kore na — stdin
// EOF-e readline close hoye jay ar 2nd call ERR_USE_AFTER_CLOSE throw kore.
// Tai non-TTY te async-iterator path ta use kori, jate script kora jay.
async function* inputLines() {
  if (!process.stdin.isTTY) {
    for await (const line of rl) yield line;
    return;
  }
  while (true) {
    try {
      yield await rl.question(green("\n› "));
    } catch (err) {
      if (err.code === "ERR_USE_AFTER_CLOSE") return;
      throw err;
    }
  }
}

/** @type {import("langchain").BaseMessage[]} */
let history = [];
let lastResult = null;

/** Turn chalao ar history ta agent-er ferot deoya full message list diye replace koro. */
async function run(agent, userMessage, { stream = true } = {}) {
  const input = { messages: [...history, userMessage] };

  const result = stream
    ? await streamRun(agent, input, runConfig)
    : await agent.invoke(input, runConfig);

  history = result.messages;
  lastResult = result;
  return result;
}

try {
  for await (const raw of inputLines()) {
    const line = raw.trim();
    if (!line) continue;
    if (line === "q") break;
    if (!process.stdin.isTTY) console.log(green(`\n› ${line}`));

    if (line === "/help") {
      console.log(HELP);
      continue;
    }

    if (line === "/reset") {
      history = [];
      lastResult = null;
      console.log(dim("history cleared"));
      continue;
    }

    // [T6] Artifact — tool er 2nd tuple element. Model eta kokhono dekhe na,
    // shudhu tomar code dekhe. UI te table render korar jonno perfect.
    if (line === "/artifacts") {
      const arts = collectArtifacts(lastResult?.messages ?? []);
      console.log(arts.length ? JSON.stringify(arts, null, 2) : dim("no artifacts"));
      continue;
    }

    // [T7] contentBlocks — content string ba provider-native array jai hok,
    // ei getter shob time normalised typed block dey.
    if (line === "/blocks") {
      const last = lastResult?.messages?.at(-1);
      console.log(last ? JSON.stringify(last.contentBlocks, null, 2) : dim("nothing yet"));
      continue;
    }

    // [T3] Structured output — result.structuredResponse.
    if (line.startsWith("/json ")) {
      const result = await run(
        agents.structured,
        new HumanMessage(line.slice(6)),
        { stream: false },
      );
      console.log(JSON.stringify(result.structuredResponse, null, 2));
      continue;
    }

    // [T4] runtime.writer theme asha custom event.
    if (line.startsWith("/custom ")) {
      await streamCustomEvents(
        agents.chat,
        { messages: [...history, new HumanMessage(line.slice(8))] },
        runConfig,
      );
      continue;
    }

    // [T7] Multimodal — contentBlocks diye text + image ek message e.
    // Gotcha: JS-e camelCase (`mimeType`, `fileId`). Docs-er kichu example e
    // Python-flavoured `source_type` / `mime_type` ache — oita chupchap drop hoy.
    if (line.startsWith("/image ")) {
      const [url, question = "Describe this image."] = line
        .slice(7)
        .split("|")
        .map((s) => s.trim());

      await run(
        agents.chat,
        new HumanMessage({
          contentBlocks: [
            { type: "text", text: question },
            { type: "image", url },
            // base64 hole: { type: "image", data: "<b64>", mimeType: "image/jpeg" }
          ],
        }),
      );
      console.log();
      continue;
    }

    const result = await run(agents.chat, new HumanMessage(line));
    console.log();

    const usage = formatUsage(result.messages.at(-1));
    if (usage) console.log(dim(`  ${usage}`));
  }
} finally {
  rl.close();
  await closeMcp();
}
