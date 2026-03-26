/**
 * 使用LCEL，可以构造出结构最简单的Chain。
 * LangChain表达式语言（LCEL，LangChain Expression Language）是一种声明式方法，可以轻松地
 * 将多个组件链接成 AI 工作流。通过 .pipe() 将组件连接成可执行流程。
 * LCEL的基本构成：提示（Prompt）+ 模型（Model）+ 输出解析器（OutputParser）
 */

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.openai-proxy.org/v1" },
  modelName: "gpt-4o-mini",
});

// ─────────────────────────────────────────────
// 1. 最基础的 pipe：Prompt → Model → Parser
// ─────────────────────────────────────────────
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个翻译专家，请将用户的话翻译成{language}。"],
  ["human", "{text}"],
]);

const basicChain = prompt
  .pipe(model)
  .pipe(new StringOutputParser());

const result1 = await basicChain.invoke({
  language: "英文",
  text: "今天天气真不错，适合出去走走。",
});
console.log("1. 基础 pipe 结果：", result1);


// ─────────────────────────────────────────────
// 2. pipe + RunnableLambda：在链中插入自定义函数
// ─────────────────────────────────────────────
const upperCase = RunnableLambda.from((text: string) => text.toUpperCase());

const transformChain = prompt
  .pipe(model)
  .pipe(new StringOutputParser())
  .pipe(upperCase);               // 把输出转成大写

const result2 = await transformChain.invoke({
  language: "英文",
  text: "你好，世界！",
});
console.log("2. 带自定义函数的 pipe：", result2);


// ─────────────────────────────────────────────
// 3. RunnableSequence：等价于多个 .pipe()，语义更清晰
// ─────────────────────────────────────────────
const summaryPrompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个文章摘要助手，用{words}个字以内总结用户输入。"],
  ["human", "{article}"],
]);

const sequenceChain = RunnableSequence.from([
  summaryPrompt,
  model,
  new StringOutputParser(),
]);

const result3 = await sequenceChain.invoke({
  words: "30",
  article: "LangChain 是一个用于构建大语言模型应用的框架，它提供了链、代理、记忆等核心组件，帮助开发者快速构建 AI 应用。",
});
console.log("3. RunnableSequence 结果：", result3);


// ─────────────────────────────────────────────
// 4. 流式输出（stream）
// ─────────────────────────────────────────────
console.log("\n4. 流式输出：");
const streamChain = prompt.pipe(model).pipe(new StringOutputParser());
const stream = await streamChain.stream({
  language: "日文",
  text: "我喜欢编程，每天都在学习新技术。",
});
for await (const chunk of stream) {
  process.stdout.write(chunk);
}
console.log();
