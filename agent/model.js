// [T5] Models — ChatOpenRouter, params, token usage.
//
// Gotcha: initChatModel("openrouter:...") kaj kore NA. `openrouter` LangChain-er
// MODEL_PROVIDER_CONFIG e nei, tai class ta direct construct korte hobe.
// (initChatModel er valid prefix: openai, anthropic, google, groq, mistralai,
// ollama, bedrock, xai, deepseek, together, fireworks, cerebras, perplexity...)
import { ChatOpenRouter } from "@langchain/openrouter";

export const MAIN_MODEL = "anthropic/claude-sonnet-4.5";
export const CHEAP_MODEL = "anthropic/claude-haiku-4.5";

/** @param {{ model?: string, temperature?: number, maxTokens?: number }} [opts] */
export function makeModel(opts = {}) {
  return new ChatOpenRouter({
    model: opts.model ?? MAIN_MODEL,
    apiKey: process.env.OPENROUTER_API_KEY,
    temperature: opts.temperature ?? 0,
    maxTokens: opts.maxTokens ?? 2048,

    // OpenRouter-specific — ChatOpenAI + baseUrl diye korle ei field gula haray.
    siteName: "langchain-v1-demo",
    provider: { allow_fallbacks: true, data_collection: "deny" },

    // Jei param LangChain surface kore na (e.g. reasoning), ta modelKwargs
    // diye pathaw — eta request body te spread hoy.
    // modelKwargs: { reasoning: { effort: "high" } },
  });
}

// [T5] Token usage — AIMessage.usage_metadata. Note: usage_metadata /
// tool_calls / response_metadata snake_case, kintu .text / .contentBlocks
// camelCase. Ei mixed casing ta v1-e shobcheye beshi bhul hoy.
/** @param {import("langchain").AIMessage} msg */
export function formatUsage(msg) {
  const u = msg?.usage_metadata;
  if (!u) return "";
  const cached = u.input_token_details?.cache_read ?? 0;
  const reasoning = u.output_token_details?.reasoning ?? 0;
  return [
    `in ${u.input_tokens}`,
    `out ${u.output_tokens}`,
    cached ? `cached ${cached}` : null,
    reasoning ? `reasoning ${reasoning}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}
