import { PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, FewShotPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config"; // 加载配置文件 .env
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({ 
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
})

// 1. 定义模板：就像定义一个 React 组件
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "你是一个专业的{topic}翻译官，负责把中文翻译成{language}。"],
  ["human", "请翻译这段话：{text}"]
]);

// 2. 传入变量：就像传 Props
const formattedPrompt = await promptTemplate.invoke({
  topic: "前端技术",
  language: "英文",
  text: "这个组件的性能需要优化。",
});

// console.log(formattedPrompt);
// console.log(formattedPrompt.toChatMessages());

const response = await model.invoke(formattedPrompt);  // Model processes the result
// console.log(response.content); // 直接查看格式化后的消息数组，看看系统和用户消息是如何组织的


// --------------------------------- MessagesPlaceholder ----------------------------------
// 1. 定义 Prompt 模板
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个幽默的翻译助手。"],
  // 关键：定义一个名为 "chat_history" 的占位符
  new MessagesPlaceholder("chat_history"), 
  ["human", "{input}"],
]);

// 2. 模拟对话历史 (在实际 Agent 中，这通常来自 State)
const history = [
  new HumanMessage("你好，我叫 Savannah。"),
  new AIMessage("你好 Savannah！我是你的翻译小助手，有什么可以帮你的？"),
];

// 4. 组合并调用
const chain = prompt.pipe(model);

const response1 = await chain.invoke({
  chat_history: history, // 将数组注入占位符
  input: "我刚才说我叫什么名字？用英文翻译并回答。",
});

console.log(response1);


// ------- FewShotPromptTemplate 与PromptTemplate一起使用
// 1. 准备“葫芦” (Examples)
const examples = [
  {"input": "北京天气怎么样", "output": "北京市"},
  {"input": "南京下雨吗", "output": "南京市"},
  {"input": "武汉热吗", "output": "武汉市"}
];

// 2. 规定“葫芦”怎么摆 (Example Prompt)
// 这里的变量名必须对应 examples 里的键名
const examplePrompt = new PromptTemplate({
  inputVariables: ["input", "output"],
  template: "用户提问: {input}\n专业回答: {output}",
});

// 3. 组合成最终模板 (FewShotPromptTemplate)
const dynamicPrompt = new FewShotPromptTemplate({
  examples: examples, // 少量的人工示例（dict 列表）
  examplePrompt: examplePrompt, // 如何格式化每个示例（使用 PromptTemplate）
  prefix: "你是一个天气播报员", // 示例之前的文字说明
  suffix: "用户提问: {input}\n专业回答:", // 用户真正的问题模板
  inputVariables: ["input"],  //最终 suffix 中需要传入的变量
})

// 4. 渲染测试
const prompt1 = await dynamicPrompt.invoke({ input: "上海现在热吗" });
// console.log(prompt1);

const response2 = await model.invoke(prompt1);
// console.log(response2.content);


// ------- FewShotChatMessagePromptTemplate 与 ChatPromptTemplate 一起使用
// 自动将示例格式化为聊天消息（ HumanMessage / AIMessage 等）
// 输出结构化聊天消息（ List[BaseMessage] ）
// 保留对话轮次结构

//  1.示例消息格式
const examples1 = [
  {"input": "1+1等于几？", "output": "1+1等于2"},
  {"input": "法国的首都是？", "output": "巴黎"}
]

// 2.定义示例的消息格式提示词模版
const msgExamplePrompt = ChatPromptTemplate.fromMessages([
  ["human", "{input}"],
  ["ai", "{output}"],
]);

// 3.定义FewShotChatMessagePromptTemplate对象
const fewShotPrompt = new FewShotChatMessagePromptTemplate({
  examplePrompt: msgExamplePrompt,
  examples: examples1,
  inputVariables: [], // 示例是静态的，无需外部变量
});
// console.log(fewShotPrompt);

// 4. 拼装成最终的对话模板
const finalPrompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个数学奇才"],
  fewShotPrompt, // ⬅️ 例子插在这里
  ["human", "{input}"],
]);

const response3 = await finalPrompt.pipe(model).invoke({ input: "请问123+143等于几？" });
console.log(response3.content);

// ------------ Example selectors  --------------
// 前面FewShotPromptTemplate的特点是，无论输入什么问题，都会包含全部示例。在实际开发中，我们可以根据当前输入，使用示例选择器，从大量候选示例中选取最相关的示例子集。

// 使用的好处：避免盲目传递所有示例，减少 token 消耗的同时，还可以提升输出效果。

// 示例选择策略：语义相似选择、长度选择、最大边际相关示例选择等

// 语义相似选择：通过余弦相似度等度量方式评估语义相关性，选择与输入问题最相似的 k 个示例。
// 长度选择：根据输入文本的长度，从候选示例中筛选出长度最匹配的示例。增强模型对文本结构的理解。比语义相似度计算更轻量，适合对响应速度要求高的场景。
// 最大边际相关示例选择：优先选择与输入问题语义相似的示例；同时，通过惩罚机制避免返回同质化的内容。
