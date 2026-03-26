
/**
 * Langchain 调用本地 Ollama 模型
 * 前置步骤：
 * 1. 安装 ollama：https://ollama.com
 * 2. 下载模型：ollama pull deepseek-r1:7b
 * 3. 启动服务：ollama serve（默认监听 http://localhost:11434）
 */

import "dotenv/config";
import { ChatOllama } from "@langchain/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// 初始化本地 Ollama 模型
const model = new ChatOllama({
  baseUrl: "http://localhost:11434", // Ollama 默认地址 若 Ollama 不在本地默认端口运行，需指定 base_url ，即：
  model: "deepseek-r1:7b",          // 换成你已下载的模型名
  temperature: 0.7,
});

// ------------ 1. 直接调用 ------------
const response = await model.invoke("用一句话解释什么是向量数据库");
console.log("直接调用结果：", response.content);

// ------------ 2. 流式输出 ------------
console.log("\n流式输出：");
const stream = await model.stream("给我讲个笑话");
for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
console.log();

// ------------ 3. Prompt + Chain ------------
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个专业的{role}，请用简洁的语言回答。"],
  ["human", "{question}"],
]);

const chain = prompt.pipe(model).pipe(new StringOutputParser());

const result = await chain.invoke({
  role: "TypeScript 开发者",
  question: "pipe() 和 compose() 有什么区别？",
});
console.log("\nChain 调用结果：", result);
