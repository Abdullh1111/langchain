// [T4] Streaming — streamEvents v3 typed projections.
//
// Purono way: agent.stream(input, { streamMode: "messages" | "updates" | "values"
// | "custom" | "tools" | "debug" }) → tuple chunk. Ekhono kaj kore.
//
// Notun way (v1.3+): agent.streamEvents(input, { version: "v3" }) → ekta
// AgentRunStream jekhane alada alada TYPED iterator ache:
//   stream.messages    → proti LLM call; .text / .reasoning / .toolCalls / .usage / .output
//   stream.toolCalls   → tool EXECUTION lifecycle; .name .input .status await .output
//   stream.values      → state snapshot
//   stream.output      → final state (await)
//   stream.subagents   → named subagent runs, nested
//   stream.middleware  → middleware lifecycle events
//
// ⚠️ JS-e projection gula CONCURRENTLY consume korte hobe (Promise.all).
// Sequential 2nd `for await` starve kore — eta shobcheye common bug.
const dim = (s) => `\x1b[2m${s}\x1b[0m`;
const cyan = (s) => `\x1b[36m${s}\x1b[0m`;
const yellow = (s) => `\x1b[33m${s}\x1b[0m`;

/**
 * @param {import("langchain").ReactAgent} agent
 * @param {object} input
 * @param {object} config
 */
export async function streamRun(agent, input, config) {
  const stream = await agent.streamEvents(input, { ...config, version: "v3" });

  await Promise.all([
    // 1. Model output — reasoning + text token by token
    (async () => {
      for await (const message of stream.messages) {
        for await (const delta of message.reasoning) {
          process.stdout.write(dim(delta));
        }
        for await (const delta of message.text) {
          process.stdout.write(delta);
        }
      }
    })(),

    // 2. Tool execution lifecycle
    (async () => {
      for await (const call of stream.toolCalls) {
        process.stdout.write(
          `\n${cyan(`⚙ ${call.name}`)} ${dim(JSON.stringify(call.input))}\n`,
        );
        const output = await call.output;
        const text = typeof output === "string" ? output : JSON.stringify(output);
        process.stdout.write(dim(`  ↳ ${text.slice(0, 160)}\n`));
      }
    })(),

    // 3. Subagent runs — nested, tai depth boojha jay
    (async () => {
      for await (const sub of stream.subagents) {
        process.stdout.write(`\n${yellow(`↳ subagent: ${sub.name}`)}\n`);
      }
    })(),
  ]);

  // Final state — messages, structuredResponse, middleware state shob ekhane.
  return await stream.output;
}

/**
 * [T4] Purono streamMode API — "custom" event (runtime.writer theke) dekhar
 * jonno eta ekhono shob theke shoja way.
 */
export async function streamCustomEvents(agent, input, config) {
  const stream = await agent.stream(input, {
    ...config,
    streamMode: ["updates", "custom"],
  });

  for await (const [mode, chunk] of stream) {
    if (mode === "custom") {
      console.log(dim(`  ⋯ ${JSON.stringify(chunk)}`));
    } else {
      console.log(dim(`  [${Object.keys(chunk).join(",")}]`));
    }
  }
}
