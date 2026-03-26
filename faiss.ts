import "dotenv/config";
import { SemanticSimilarityExampleSelector } from "@langchain/core/example_selectors";
import { FakeEmbeddings } from "@langchain/core/utils/testing";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { FewShotPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

// ── 候选示例库 ────────────────────────────────────────────────────────────
const examples = [
  { input: "苹果",   output: "水果" },
  { input: "猫",     output: "动物" },
  { input: "玫瑰",   output: "植物" },
  { input: "西瓜",   output: "水果" },
  { input: "老虎",   output: "动物" },
  { input: "松树",   output: "植物" },
  { input: "草莓",   output: "水果" },
  { input: "鲨鱼",   output: "动物" },
];

// ── 语义相似度选择器（基于 FAISS 向量数据库）─────────────────────────────
// FaissStore：Facebook 开源的高性能向量检索库，支持百万级向量快速搜索
// FakeEmbeddings：随机向量，用于演示流程（替换为真实 embedding 可获得真语义）
const exampleSelector = await SemanticSimilarityExampleSelector.fromExamples(
  examples,
  new FakeEmbeddings(),
  FaissStore,
  { k: 2 } // 每次选出最相关的 2 条示例
);

// ── 单条示例的格式模板 ────────────────────────────────────────────────────
const examplePrompt = new PromptTemplate({
  inputVariables: ["input", "output"],
  template: "输入：{input}\n输出：{output}",
});

// ── FewShotPromptTemplate：动态拼装少样本提示词 ───────────────────────────
const fewShotPrompt = new FewShotPromptTemplate({
  exampleSelector,           // 动态选择示例，而非写死 examples 数组
  examplePrompt,             // 每条示例的格式
  prefix: "请根据示例，对以下输入进行分类：\n\n示例：",
  suffix: "\n输入：{input}\n输出：",
  inputVariables: ["input"],
});

// ── 模型 ──────────────────────────────────────────────────────────────────
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
  temperature: 0,
});

// ── 运行 ──────────────────────────────────────────────────────────────────
const question = "芒果";

// 查看 FAISS 选出了哪些示例
const selected = await exampleSelector.selectExamples({ input: question });
console.log(`📌 FAISS 为「${question}」选出的示例：`);
selected.forEach((ex, i) => console.log(`  [${i + 1}] ${ex.input} -> ${ex.output}`));

// 查看最终渲染的提示词
const rendered = await fewShotPrompt.format({ input: question });
console.log("\n📝 最终提示词：\n");
console.log(rendered);

// 调用模型
const chain = fewShotPrompt.pipe(model);
const response = await chain.invoke({ input: question });
console.log("🤖 模型分类结果：", response.content);
