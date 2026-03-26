/**
 * Memory —— LangChain 多轮对话上下文管理
 *
 * 核心思路：每次对话前把历史消息注入 Prompt，让模型"记住"之前说了什么。
 *
 * 本文件演示三种方式：
 * 1. 手动管理历史（最底层，理解原理）
 * 2. ChatMessageHistory（LangChain 内置消息列表）
 * 3. RunnableWithMessageHistory（LCEL 链自动管理 memory，推荐用法）
 */

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.openai-proxy.org/v1" },
  modelName: "gpt-4o-mini",
});

// ─────────────────────────────────────────────
// 1. 手动管理历史消息（理解底层原理）
// ─────────────────────────────────────────────
console.log("=== 1. 手动管理历史 ===");

const history: (HumanMessage | AIMessage)[] = [];

async function chatManual(userInput: string) {
  history.push(new HumanMessage(userInput));

  const response = await model.invoke([
    new AIMessage("你是一个友好的助手，记住用户说的所有信息。"),
    ...history,
  ]);

  history.push(new AIMessage(response.content as string));
  return response.content;
}

console.log(await chatManual("我叫 Savannah，我喜欢打篮球。"));
console.log(await chatManual("我刚才说我叫什么？喜欢什么运动？"));


// ─────────────────────────────────────────────
// 2. InMemoryChatMessageHistory（LangChain 内置）
// ─────────────────────────────────────────────
console.log("\n=== 2. InMemoryChatMessageHistory ===");

const messageHistory = new InMemoryChatMessageHistory();

async function chatWithHistory(userInput: string) {
  await messageHistory.addMessage(new HumanMessage(userInput));

  const messages = await messageHistory.getMessages();
  const response = await model.invoke(messages);

  await messageHistory.addMessage(new AIMessage(response.content as string));
  return response.content;
}

console.log(await chatWithHistory("我是一名前端开发者，主要用 TypeScript。"));
console.log(await chatWithHistory("根据我的背景，推荐我学习什么 AI 框架？"));


// ─────────────────────────────────────────────
// 3. RunnableWithMessageHistory（LCEL 自动管理，推荐）
//    支持多用户会话隔离（不同 sessionId 独立记忆）
// ─────────────────────────────────────────────
console.log("\n=== 3. RunnableWithMessageHistory（多用户隔离）===");

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个智能助手，请根据对话历史回答问题。"],
  new MessagesPlaceholder("chat_history"), // 历史消息注入点
  ["human", "{input}"],
]);

const chain = prompt.pipe(model).pipe(new StringOutputParser());

// 用 Map 存储每个用户的会话历史
const sessionStore = new Map<string, InMemoryChatMessageHistory>();

const chainWithMemory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: (sessionId: string) => {
    if (!sessionStore.has(sessionId)) {
      sessionStore.set(sessionId, new InMemoryChatMessageHistory());
    }
    return sessionStore.get(sessionId)!;
  },
  inputMessagesKey: "input",
  historyMessagesKey: "chat_history",
});

// 模拟用户 A 的对话
const configA = { configurable: { sessionId: "user_A" } };
console.log("用户A：", await chainWithMemory.invoke({ input: "我叫小明，我在学 LangChain。" }, configA));
console.log("用户A：", await chainWithMemory.invoke({ input: "我之前说我在学什么？" }, configA));

// 模拟用户 B（与 A 的记忆完全隔离）
const configB = { configurable: { sessionId: "user_B" } };
console.log("用户B：", await chainWithMemory.invoke({ input: "我叫什么？" }, configB));
// 用户B没有说过名字，所以模型不知道
