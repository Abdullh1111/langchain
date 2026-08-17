// [T11] MCP — MCP server theke tool load kore createAgent e dhukanor pattern.
//
// Opt-in: `ENABLE_MCP=1` set korle chalu hoy, karon network lage.
// LangChain-er nijer public MCP server ta free ar key lagena — test korar
// jonno perfect.
//
// Gotcha: TS adapter-e tool error hole `ToolException` THROW kore (Python
// adapter-er moto model ke error ToolMessage return kore na). Tai try/catch
// na dile agent loop-i mara jabe.
import { MultiServerMCPClient } from "@langchain/mcp-adapters";

let client;

export async function loadMcpTools() {
  if (process.env.ENABLE_MCP !== "1") return { tools: [], reason: "ENABLE_MCP != 1" };

  client = new MultiServerMCPClient({
    // http transport — remote server
    langchainDocs: {
      transport: "http",
      url: "https://docs.langchain.com/mcp",
    },
    // stdio transport — local subprocess. Uncomment korte chaile:
    // math: { transport: "stdio", command: "node", args: ["/abs/path/math_server.js"] },
  });

  try {
    const tools = await client.getTools();
    return { tools, reason: null };
  } catch (err) {
    // MCP down thakleo agent ta cholte hobe — degrade koro, crash na.
    return { tools: [], reason: `MCP load failed: ${err.message}` };
  }
}

export async function closeMcp() {
  await client?.close();
}
