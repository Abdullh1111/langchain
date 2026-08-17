// [T3] Structured output.
//
// 3 ta strategy:
//   responseFormat: Schema          → auto (model support korle provider, noile tool)
//   providerStrategy(Schema)        → provider-native JSON mode (support na thakle tool e fallback)
//   toolStrategy(Schema, options)   → force tool calling, error handling niye
//
// v1-er boro win: structured output ekhon MAIN LOOP e generate hoy — age
// er moto extra ekta LLM call lage na.
//
// Parsed object always `result.structuredResponse` e boshe. responseFormat
// na dile ei key ta state-e thake-i na.
import { toolStrategy, StructuredOutputParsingError } from "langchain";
import * as z from "zod";

export const Reply = z.object({
  answer: z.string().describe("The answer to show the user, in plain language"),
  confidence: z
    .enum(["high", "medium", "low"])
    .describe("low if the handbook did not cover the question"),
  sources: z
    .array(z.string())
    .describe("Policy names used, e.g. leave-policy. Empty array if none."),
  needsHuman: z.boolean().describe("true if a human should follow up"),
});

// PRODUCTION e bare schema pathacchi (`responseFormat: Reply`) — eta auto
// select kore: model native structured output support korle providerStrategy,
// noile toolStrategy e fallback. Docs-er recommended default.
export const replyFormat = Reply;

// Force tool-calling korte chaile eta. handleError diye validation fail hole
// model ke retry korano jay.
//
// ⚠️ Ekhane ekta ashol gotcha ache: toolStrategy nijer tool-er name AUTO
// generate kore ekta global counter diye — `extract-1`, `extract-2`...
// Mane (a) model ekta meaningless tool name dekhe, ar (b) test-e tumi
// name ta predict korte parbe na. Eijonyoi production e bare schema
// bhalo, ar unit test e providerStrategy use kora shohoj.
export const forcedToolFormat = toolStrategy(Reply, {
  // Default behaviour: throw kore NA — error ta ToolMessage hishebe model ke
  // ferot dey ar model abar try kore. handleError: false dile propagate korbe.
  handleError: (error) => {
    if (error instanceof StructuredOutputParsingError) {
      return `Your structured response was invalid (${error.errors.join("; ")}). Fix it and reply again.`;
    }
    return error.message;
  },
});
