import { config } from "dotenv";
import readline from "readline/promises";

config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("api key:", process.env.OPENROUTER_API_KEY);

while (true) {
  const question = await rl.question("Ask a question (q to quit): ");

  if (question.toLowerCase() === "q") {
    break;
  }

  console.log("question:", question);
}

rl.close();