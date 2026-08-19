// Step 1 — shobcheye simple agent. 5 ta jinis, er beshi kichu na.
// Step 2 — systemPrompt + terminal loop + history.
// Step 3 — streaming (invoke → streamEvents).
import { ChatOpenRouter } from "@langchain/openrouter";
import { config } from "dotenv";
import { createAgent, tool } from "langchain";
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
const getWeather = tool(
  ({ city }) => {
    console.log("weather tool called");
    return `It's 24°C and sunny in ${city}.`;
  },
  {
    name: "get_weather",
    description: "Get the current weather in a given location",
    schema: z.object({
      city: z.string().describe("The city to get the weather for"),
    }),
  },
);

// 4. agent — `new` na, plain function call
const agent = createAgent({
  model,
  tools: [getWeather],

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
async function streamRun(input) {
  const stream = await agent.streamEvents(input, { version: "v3" });

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
        process.stdout.write(`[tool] ↳ ${await call.output}\n`);
      }
    })(),
  ]);

  // Stream sesh — final state. invoke() er return value-r shathe ek.
  return await stream.output;
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
}

rl.close();
