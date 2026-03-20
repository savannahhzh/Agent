import "dotenv/config";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// 1. 定义一个工具 (模拟搜索最新技术文档)
const searchDocs = tool(
  async ({ query }) => {
    console.log(`🔍 [Tool] 正在搜索库中关于 "${query}" 的文档...`);
    // 模拟从数据库或网页抓取的结果
    if (query.includes("LangGraph")) {
      return "2026年最新：LangGraph 支持分布式多智能体协作，并推出了可视化 Studio 2.0。";
    }
    return "未找到相关技术文档。";
  },
  {
    name: "search_tech_docs",
    description: "当用户询问有关 LangGraph 或最新技术动态时，调用此工具获取最新信息。",
    schema: z.object({
      query: z.string().describe("搜索关键词"),
    }),
  }
);

// 2. 将工具绑定到 DeepSeek (注意：DeepSeek 完美支持 Tool Use)
const tools = [searchDocs];
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
}).bindTools(tools); // ⬅️ 关键：把工具挂载到模型上

// 3. 定义 State (增加一个消息列表来存放对话)
const AgentState = Annotation.Root({
  messages: Annotation<any[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

// 4. 节点逻辑
const callModel = async (state: typeof AgentState.State) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

const toolNode = async (state: typeof AgentState.State) => {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage.tool_calls;
  
  const results = await Promise.all(
    toolCalls.map(async (tc: any) => {
      const result = await searchDocs.invoke(tc.args);
      return {
        role: "tool",
        content: result,
        tool_call_id: tc.id,
      };
    })
  );
  return { messages: results };
};

// 5. 构建图：这里引入了决策分支
const workflow = new StateGraph(AgentState)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const lastMessage = state.messages[state.messages.length - 1];
    // 如果 AI 决定调用工具，就走 tools 节点，否则结束
    return lastMessage.tool_calls?.length > 0 ? "tools" : END;
  })
  .addEdge("tools", "agent"); // 工具执行完后，必须回到 agent 让它总结输出

const app = workflow.compile();

// 6. 运行测试
const final = await app.invoke({ 
  messages: [{ role: "user", content: "2026年的 LangGraph 有什么新特性？" }] 
});

console.log("\n🤖 AI 的最终回复：\n", final.messages[final.messages.length - 1].content);