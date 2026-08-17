// [T2] Middleware — built-in stack + custom hooks.
// [T8] Context engineering — summarization + context editing.
// [T9] Guardrails — PII + input filter + output check.
import {
  AIMessage,
  ClearToolUsesEdit,
  contextEditingMiddleware,
  createMiddleware,
  dynamicSystemPromptMiddleware,
  llmToolSelectorMiddleware,
  modelCallLimitMiddleware,
  piiMiddleware,
  summarizationMiddleware,
  SystemMessage,
  toolCallLimitMiddleware,
  toolRetryMiddleware,
} from "langchain";

import { makeModel, CHEAP_MODEL } from "./model.js";
import { calculateTip } from "./tools.js";

// ─────────────────────────────────────────────────────────────────────────────
// [T2] Custom middleware — shob lifecycle hook ekta jaygay.
//
// Execution order (eta mone rakha joruri):
//   before* → first-to-last (array order)
//   after*  → last-to-first (reverse)
//   wrap*   → nested; prothom middleware baki shob ke wrap kore
// Tai critical middleware array-r SHURUTE rakho.
// ─────────────────────────────────────────────────────────────────────────────
export function traceMiddleware({ verbose = true } = {}) {
  let modelCalls = 0;
  let startedAt = 0;

  // stderr e likhchi, stdout e na — noile streaming token gula ar trace line
  // gula ek shathe mishe output ta porar ojoggo hoye jay. `2>/dev/null` diye
  // trace off korte paro.
  const log = (...a) => verbose && console.error("  \x1b[2m│", ...a, "\x1b[0m");

  return createMiddleware({
    name: "TraceMiddleware",

    // once per invoke, shob kichur age
    beforeAgent: () => {
      modelCalls = 0;
      startedAt = Date.now();
      log("beforeAgent");
    },

    // prottek model call er age
    beforeModel: (state) => {
      log(`beforeModel #${modelCalls + 1} · messages=${state.messages.length}`);
    },

    // model call ke wrap kore — request ta ekhane MODIFY korte paro
    wrapModelCall: async (request, handler) => {
      log(`wrapModelCall · tools=${request.tools.length}`);

      // System prompt e runtime info add kori. `.concat()` use korchi, string
      // assign na — noile onno middleware-er cache-control block gula haray.
      const response = await handler({
        ...request,
        systemMessage: request.systemMessage.concat(
          new SystemMessage(
            `\n<runtime>Turn ${modelCalls + 1} of this run.</runtime>`,
          ),
        ),
      });

      log(`wrapModelCall done · toolCalls=${response.tool_calls?.length ?? 0}`);
      return response;
    },

    // prottek tool call ke wrap kore — timing, retry, audit log
    wrapToolCall: async (request, handler) => {
      const t0 = Date.now();
      log(`tool → ${request.toolCall.name}(${JSON.stringify(request.toolCall.args)})`);
      try {
        const result = await handler(request);
        log(`tool ← ${request.toolCall.name} ok in ${Date.now() - t0}ms`);
        return result;
      } catch (err) {
        log(`tool ✗ ${request.toolCall.name}: ${err.message}`);
        throw err;
      }
    },

    afterModel: () => {
      modelCalls += 1;
    },

    afterAgent: () => {
      log(`afterAgent · modelCalls=${modelCalls} · ${Date.now() - startedAt}ms`);
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// [T6] Runtime-e tool register kora.
// wrapModelCall → model ke tool ta dekhay.  wrapToolCall → execute korte dey.
// DUITAI lage. shudhu wrapModelCall dile model tool call korbe kintu execute
// korte parbe na.
// ─────────────────────────────────────────────────────────────────────────────
export const dynamicToolsMiddleware = createMiddleware({
  name: "DynamicToolsMiddleware",

  wrapModelCall: (request, handler) => {
    // Real life e eta DB / MCP registry / per-company config theke ashbe.
    const discovered = [calculateTip];
    return handler({ ...request, tools: [...request.tools, ...discovered] });
  },

  wrapToolCall: (request, handler) => {
    if (request.toolCall.name === calculateTip.name) {
      // `tool` ta hate supply korte hobe — agent-er static list e ei tool nei.
      return handler({ ...request, tool: calculateTip });
    }
    return handler(request);
  },
});

// ─────────────────────────────────────────────────────────────────────────────
// [T9] Guardrail 1 — deterministic input filter, model call korar AGE.
// jumpTo:"end" diye short-circuit. `canJumpTo` declare korte HOBE.
// ─────────────────────────────────────────────────────────────────────────────
export function inputGuardMiddleware(bannedPhrases) {
  const banned = bannedPhrases.map((p) => p.toLowerCase());

  return createMiddleware({
    name: "InputGuardMiddleware",
    beforeAgent: {
      canJumpTo: ["end"],
      hook: (state) => {
        const last = state.messages?.at(-1);
        if (!last || last.getType?.() !== "human") return;

        const text = (last.text ?? "").toLowerCase();
        const hit = banned.find((p) => text.includes(p));
        if (!hit) return;

        return {
          messages: [
            new AIMessage(
              "I can't help with that request. Try rephrasing it as a work question.",
            ),
          ],
          jumpTo: "end",
        };
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// [T9] Guardrail 2 — model-based output check, uttor user ke deyar age.
// LLM-as-judge. Cheap model use koro, noile proti turn e double cost.
// ─────────────────────────────────────────────────────────────────────────────
export function outputGuardMiddleware() {
  const judge = makeModel({ model: CHEAP_MODEL, maxTokens: 8 });

  return createMiddleware({
    name: "OutputGuardMiddleware",
    afterAgent: {
      canJumpTo: ["end"],
      hook: async (state) => {
        const last = state.messages?.at(-1);
        if (!last || last.getType?.() !== "ai") return;

        const text = last.text ?? "";
        if (!text.trim()) return;

        const verdict = await judge.invoke([
          {
            role: "user",
            content:
              "Does this assistant reply leak a raw API key, password, or database " +
              'connection string? Answer with exactly one word: SAFE or UNSAFE.\n\n' +
              `Reply:\n${text.slice(0, 2000)}`,
          },
        ]);

        if (!(verdict.text ?? "").toUpperCase().includes("UNSAFE")) return;

        return {
          messages: [
            new AIMessage("I withheld that response because it contained a secret."),
          ],
          jumpTo: "end",
        };
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Puro stack. Order-i design — critical jinis age.
// ─────────────────────────────────────────────────────────────────────────────
export function buildMiddleware({ verbose = true } = {}) {
  return [
    // 1. Guardrail age — banned input model porjonto pouchabei na.
    inputGuardMiddleware(["ignore previous instructions", "jailbreak"]),

    // 2. [T9] PII — user input theke email/card redact kore model e pathay.
    //    Signature: piiMiddleware(type, options) — type POSITIONAL arg.
    //    NOTE: piiRedactionMiddleware deprecated, ar docs-er guardrails page
    //    ekhono oi purono form-e example dey. piiMiddleware use koro.
    piiMiddleware("email", { strategy: "redact", applyToInput: true }),
    piiMiddleware("credit_card", { strategy: "mask", applyToInput: true }),

    // 3. Budget — tomar CRM-er hate-lekha step-budget er official version.
    //    threadLimit-er jonno checkpointer lage (= LangGraph), tai runLimit.
    modelCallLimitMiddleware({ runLimit: 12, exitBehavior: "end" }),
    toolCallLimitMiddleware({ runLimit: 20, exitBehavior: "continue" }),

    // 4. Trace — er niche ja ache shob wrap kore.
    traceMiddleware({ verbose }),

    // 5. Runtime tool injection.
    dynamicToolsMiddleware,

    // 6. [T2] Dynamic system prompt — tomar learned-knowledge injection-er
    //    official version. Function ney, options object na.
    dynamicSystemPromptMiddleware((state, runtime) => {
      const { userName = "there", role = "staff" } = runtime.context ?? {};
      const lines = [
        `You are talking to ${userName} (role: ${role}).`,
        role === "admin"
          ? "They are an admin — you may discuss internal policy details freely."
          : "They are not an admin — do not speculate about unreleased policy.",
      ];
      return lines.join("\n");
    }),

    // 7. [T8] Context editing — purono tool result gula clear kore, message
    //    gula rekhe. Long thread e token blow-up thamay.
    //
    //    NOTE: 1.5.x-e config shape ta `{ trigger: {...}, keep: {...} }` —
    //    docs-e dekhano flat `{ triggerTokens, keep: 3 }` deprecated ar
    //    `keep` ekhon object, number na. `keep`-e fraction/tokens/messages
    //    er thik EKTA dite hobe, noile zod throw kore.
    contextEditingMiddleware({
      edits: [
        new ClearToolUsesEdit({
          trigger: { tokens: 20_000 },
          keep: { messages: 3 },
        }),
      ],
    }),

    // 8. [T8] Summarization — context editing-eo na kulale purono message
    //    gula summary te dhuke jay.
    summarizationMiddleware({
      model: makeModel({ model: CHEAP_MODEL }),
      trigger: { tokens: 30_000 },
      keep: { messages: 12 },
    }),

    // 9. [T2] Tool retry — flaky tool/network er jonno.
    toolRetryMiddleware({
      maxRetries: 2,
      initialDelayMs: 400,
      onFailure: "continue", // model ke error dekhao, crash na
    }),

    // 10. [T2] Tool selector — tool count barle on koro. 5-6 tool e off rakha
    //     bhalo, karon eta proti turn e ekta extra LLM call kore.
    ...(process.env.ENABLE_TOOL_SELECTOR === "1"
      ? [
          llmToolSelectorMiddleware({
            model: makeModel({ model: CHEAP_MODEL }),
            maxTools: 4,
            alwaysInclude: ["search_handbook"],
          }),
        ]
      : []),

    // 11. Output guardrail — shesh e, mane afterAgent reverse order-e
    //     shobar age chole.
    ...(process.env.ENABLE_OUTPUT_GUARD === "1" ? [outputGuardMiddleware()] : []),
  ];
}

// NOTE — modelFallbackMiddleware(...models) ekhane use korchi na: eta model
// STRING ney ar internally initChatModel diye resolve kore, kintu `openrouter`
// initChatModel-er valid prefix na. OpenRouter-e eta emnitei dorkar nei —
// makeModel() e `models: [...]` + `route: "fallback"` diye OpenRouter nijei
// fallback kore. Anthropic/OpenAI direct use korle tokhon ei middleware ta dhoro.
