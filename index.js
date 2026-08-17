import { ChatOpenRouter } from "@langchain/openrouter";
import { config } from "dotenv";
import { tool } from "langchain";
import readline from "readline/promises";
import z from "zod";

config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const get_weather = tool((input) => `It's always sunny in ${input.city}!`, {
  name: "get_weather",
  description: "Get the current weather in a given location",
  schema: z.object({
    city: z.string().describe("The city to get the weather for"),
  }),
});

const tools = [get_weather];
const toolsByName = Object.fromEntries(tools.map((t) => [t.name, t]));
console.log({ toolsByName });

const model = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  model: "anthropic/claude-sonnet-5",
}).bindTools(tools);

const messages = [];

while (true) {
  const question = await rl.question("Ask a question (q to quit): ");

  if (question.toLowerCase() === "q") {
    break;
  }

  messages.push({ role: "user", content: question });
  console.log({ messages });

  while (true) {
    const response = await model.invoke(messages);
    messages.push(response);

    if (!response.tool_calls?.length) {
      console.log(response.content);
      break;
    }

    for (const call of response.tool_calls) {
      console.log({ call });
      const result = await toolsByName[call.name].invoke(call.args);
      messages.push({
        role: "tool",
        content: String(result),
        tool_call_id: call.id,
      });
    }
  }
}

rl.close();
