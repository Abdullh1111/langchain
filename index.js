import { ChatOpenRouter } from "@langchain/openrouter";
import { config } from "dotenv";
import readline from "readline/promises";

config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("api key:", process.env.OPENROUTER_API_KEY);

const model = new ChatOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  model: "deepseek/deepseek-v4-pro",
});

while (true) {
  const question = await rl.question("Ask a question (q to quit): ");

  if (question.toLowerCase() === "q") {
    break;
  }

  const response = await model.invoke([
    {
      role: "user",
      content: question,
    },
  ]);

  console.log({ response: response.content });
}

rl.close();
