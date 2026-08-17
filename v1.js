import { createAgent, tool } from "langchain";


const get_weather = tool((input) => `It's always sunny in ${input.city}!`, {
  name: "get_weather",
  description: "Get the current weather in a given location",
  schema: z.object({
    city: z.string().describe("The city to get the weather for"),
  }),
});

const agent = new createAgent{
  tools: [get_weather],
  llm: new ChatOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY,
    model: "anthropic/claude-sonnet-5",
  }),
};