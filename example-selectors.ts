import "dotenv/config";
import { SemanticSimilarityExampleSelector } from "@langchain/core/example_selectors";
import { FakeEmbeddings } from "@langchain/core/utils/testing";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate, FewShotChatMessagePromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

// ── 候选示例库（实际场景可以有几十上百条）────────────────────────────────────
const examples = [
  {
    input: "怎么让 div 水平垂直居中？",
    output: "推荐用 Flexbox：display:flex; justify-content:center; align-items:center;",
  },
  {
    input: "CSS Grid 怎么分三列？",
    output: "grid-template-columns: repeat(3, 1fr); 即可均分三列。",
  },
  {
    input: "React useState 怎么用？",
    output: "const [count, setCount] = useState(0); 调用 setCount(n) 触发重渲染。",
  },
  {
    input: "useEffect 怎么只执行一次？",
    output: "第二个参数传空数组：useEffect(() => { ... }, []); 只在挂载时执行。",
  },
  {
    input: "怎么用 fetch 发 POST 请求？",
    output: "fetch(url, { method:'POST', body: JSON.stringify(data), headers:{'Content-Type':'application/json'} })",
  },
  {
    input: "async/await 怎么捕获错误？",
    output: "用 try/catch 包裹：try { const res = await fetch(...) } catch(e) { console.error(e) }",
  },
];

// ── 创建语义相似度选择器 ───────────────────────────────────────────────────
// FakeEmbeddings：无需 API Key，用随机向量模拟语义，适合本地开发和学习
// MemoryVectorStore：纯内存向量库，无需任何外部服务
const exampleSelector = await SemanticSimilarityExampleSelector.fromExamples(
  examples,
  new FakeEmbeddings(),
  MemoryVectorStore,
  { k: 2 } // 每次从候选库中选出最相关的 2 条
);

// ── 定义单条示例的消息格式 ────────────────────────────────────────────────
const examplePrompt = ChatPromptTemplate.fromMessages([
  ["human", "{input}"],
  ["ai", "{output}"],
]);

// ── 动态少样本模板（核心：用 exampleSelector 替代写死的 examples）──────────
const dynamicFewShotPrompt = new FewShotChatMessagePromptTemplate({
  exampleSelector,
  examplePrompt,
  inputVariables: ["input"],
});

// ── 最终对话模板 ──────────────────────────────────────────────────────────
const finalPrompt = ChatPromptTemplate.fromMessages([
  ["system", "你是一个资深前端专家，回答简洁精准，直接给出代码示例。"],
  dynamicFewShotPrompt,
  ["human", "{input}"],
]);

// ── 模型 ──────────────────────────────────────────────────────────────────
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
  temperature: 0,
});

// ── 测试：问 React Hook 相关问题，选择器应优先挑 Hook 相关示例 ───────────────
const question = "useMemo 是用来干嘛的？";

console.log("❓ 问题：", question);
console.log("\n📌 动态选出的示例：");
const selected = await exampleSelector.selectExamples({ input: question });
selected.forEach((ex, i) => console.log(`  [${i + 1}] Q: ${ex.input}`));

console.log("\n🤖 模型回答：");
const chain = finalPrompt.pipe(model);
const response = await chain.invoke({ input: question });
console.log(response.content);
