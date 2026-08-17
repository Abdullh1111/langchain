// [T10a] Custom Embeddings class.
//
// Keno lagche: OpenRouter er embeddings endpoint NEI, ar tomar .env e shudhu
// OPENROUTER_API_KEY ache. Tai ekta local deterministic embedding likhlam —
// zero API key, offline chole, RAG pipeline ta demo korar jonno jothesto.
//
// Embeddings base class-e maat 2ta abstract method:
//   embedDocuments(string[]) => Promise<number[][]>
//   embedQuery(string)       => Promise<number[]>
//
// Production e eta shorasori swap koro: OpenAIEmbeddings (@langchain/openai),
// VoyageEmbeddings / CohereEmbeddings (@langchain/community), othoba
// OllamaEmbeddings (@langchain/ollama) — local, free.
import { Embeddings } from "@langchain/core/embeddings";

const DIM = 256;

/** Hashed bag-of-words → unit vector. Semantic na, lexical overlap dhore. */
function embedOne(text) {
  const vec = new Float64Array(DIM);

  for (const token of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    // FNV-1a
    let h = 0x811c9dc5;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    // Sublinear term frequency — ekta word 50 bar thakle 50x weight na pay.
    vec[h % DIM] += 1;
  }

  for (let i = 0; i < DIM; i++) {
    if (vec[i] > 0) vec[i] = 1 + Math.log(vec[i]);
  }

  // L2 normalise — MemoryVectorStore cosine similarity kore, tai magnitude
  // matter kore na, kintu normalise na korle zero-vector e NaN ashe.
  let norm = 0;
  for (let i = 0; i < DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;

  return Array.from(vec, (v) => v / norm);
}

export class LocalEmbeddings extends Embeddings {
  constructor(fields = {}) {
    // super() lagbei — eta this.caller (retry + concurrency) set kore.
    super(fields);
  }

  async embedDocuments(texts) {
    return texts.map(embedOne);
  }

  async embedQuery(text) {
    // embedQuery ar embedDocuments-er dimension EK hote hobe. MemoryVectorStore
    // validate kore na — mismatch hole chupchap bhul score dey.
    return embedOne(text);
  }
}
