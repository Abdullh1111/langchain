// [T10b] Retrieval / RAG — splitter → vector store → retriever → tool.
//
// Gotcha: v1-e `langchain/vectorstores/memory` NEI. Legacy shob kichu
// @langchain/classic e chole geche. Splitter-o alada package.
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { LocalEmbeddings } from "./embeddings.js";

// Fake internal handbook — real project e eta DB/S3/Notion theke ashbe.
const HANDBOOK = [
  {
    source: "leave-policy",
    text: `Leave policy. Full-time staff get 20 paid annual leave days per calendar year.
Annual leave must be requested at least 14 days in advance through the HR portal.
Unused annual leave carries over to the next year, capped at 5 days.
Sick leave is 10 days per year and does not require advance notice, but a medical
certificate is required for any absence longer than 2 consecutive days.
Unpaid leave beyond these allowances needs written approval from a department head.`,
  },
  {
    source: "expense-policy",
    text: `Expense policy. Expenses under 100 AUD can be self-approved with a receipt.
Anything from 100 to 1000 AUD requires manager approval before the spend, not after.
Above 1000 AUD requires finance team sign-off and a purchase order.
Client dinners are reimbursable up to 80 AUD per head. Alcohol is not reimbursable.
Reimbursement claims must be submitted within 30 days of the expense date.`,
  },
  {
    source: "deploy-policy",
    text: `Deployment policy. Production deploys are allowed Monday through Thursday only.
Friday deploys require an incident-commander sign-off because on-call coverage is thin.
Every deploy needs a rollback plan documented in the release ticket.
Database migrations must run as a separate step before the application deploy,
never in the same step, and must be backwards compatible with the previous release.`,
  },
];

/** Ekbar index kore retriever tool ta return kore. */
export async function buildKnowledgeTool() {
  // chunkSize choto rakhlam iccha kore: boro chunk hole 3 ta policy = 3 ta
  // chunk, ar k:3 mane retriever shob kichu ferot dey — tokhon RAG-er kono
  // filtering-i hoy na. Choto chunk = ashol ranking.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 220,
    chunkOverlap: 40,
  });

  const splits = await splitter.splitDocuments(
    HANDBOOK.map(
      (d) => new Document({ pageContent: d.text, metadata: { source: d.source } }),
    ),
  );

  const store = new MemoryVectorStore(new LocalEmbeddings());
  await store.addDocuments(splits);

  // createRetrieverTool er schema FIXED — shudhu { query: string }. Nijer
  // schema chaile tool() diye hate mora, retriever.invoke() call koro.
  const tool = createRetrieverTool(store.asRetriever({ k: 3 }), {
    name: "search_handbook",
    description:
      "Search the internal company handbook (leave, expense and deployment policy). " +
      "Use this for any question about company rules instead of answering from memory.",
  });

  return { tool, store, chunks: splits.length };
}
