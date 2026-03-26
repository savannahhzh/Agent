import "dotenv/config";
import { loadPrompt } from "langchain/prompts";
import { ChatOpenAI } from "@langchain/openai";

// 从 YAML 文件加载 PromptTemplate
// loadPrompt 支持 .yaml / .json 格式，自动解析 _type / input_variables / template 字段
const prompt = await loadPrompt("./assets/prompt.yaml");

console.log("📄 加载的模板：", prompt.template);
console.log("📌 变量列表：", prompt.inputVariables);

// 格式化：传入变量，生成最终提示词字符串
const formatted = await prompt.format({ name: "小明", what: "勇敢的小兔子" });
console.log("\n📝 格式化后的提示词：\n", formatted);

// 接入模型
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
});

const chain = prompt.pipe(model);
const response = await chain.invoke({ name: "小红", what: "会飞的鱼" });
console.log("\n🤖 模型回答：\n", response.content);
