import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser, JsonOutputParser, XMLOutputParser } from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "langchain/output_parsers";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

// const model = new ChatOpenAI({
//   apiKey: process.env.DEEPSEEK_API_KEY,
//   configuration: { baseURL: "https://api.deepseek.com/v1" },
//   modelName: "deepseek-chat",
// });

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: "https://api.openai-proxy.org/v1" },
  modelName: "gpt-4o-mini",
});


const response = await model.invoke("What is the capital of France?");
// console.log(response.content);



// ------------ StringOutputParser --------------
// const parser = new StringOutputParser();
// const str_response = await parser.invoke(response);
// console.log("Parsed output:", str_response);


// ------------- JSONOutputParser --------------
// const chatPrompt = ChatPromptTemplate.fromMessages([
//   ["system", "你是一个靠谱的{role}，请以 JSON 格式回答，例如：{{\"name\": \"...\", \"dynasty\": \"...\"}}"],
//   ["human", "{question}"],
// ]);
// // 初始化解析器
// const jsonParser = new JsonOutputParser();

// // 用 pipe 链式连接：prompt -> model -> parser
// const chain = chatPrompt.pipe(model).pipe(jsonParser);
// const json_response = await chain.invoke({ 
//   role: "历史专家", 
//   question: "谁是中国的第一位皇帝？" 
// });
// console.log("Parsed JSON output:", json_response);


// 获取解析指令 (这一步很关键，它告诉 AI 应该返回什么样的 JSON 结构)
// 使用 StructuredOutputParser + Zod schema，生成明确的字段说明
// const structuredParser = StructuredOutputParser.fromZodSchema(
//   z.object({
//     name: z.string().describe("用户的姓名"),
//     age: z.number().describe("用户的年龄"),
//     skills: z.array(z.string()).describe("用户掌握的技能列表"),
//   })
// );

// const prompt1 = ChatPromptTemplate.fromMessages([
//   ["system", "你是一个信息提取助手。请根据用户提供的内容提取信息。\n{format_instructions}"],
//   ["human", "{input}"],
// ]);
// const format_instructions = structuredParser.getFormatInstructions();

// const chain1 = prompt1.pipe(model).pipe(structuredParser);

// console.log("Format instructions for the model:\n", format_instructions);
// const result = await chain1.invoke({
//   input: "我叫 Savannah，今年 25 岁，精通 TypeScript 和 React。",
//   format_instructions: format_instructions,
// }) as { name: string; age: number; skills: string[] };

// // 直接拿到 JS 对象，不再是字符串！
// console.log(result.name);   // 输出: Savannah
// console.log(result.skills); // 输出: ['TypeScript', 'React']




// --------------- XMLOutputParser ---------------
// 把模型返回的 XML 字符串自动解析成 JS 对象。
// 用户输入 → Prompt → 模型（返回 XML 字符串）→ XMLOutputParser → JS 对象

// 2. 初始化 XML 解析器
const XMLparser = new XMLOutputParser();

// 3. 创建 Prompt 模板
// 这里的 .getFormatInstructions() 会自动生成一段告诉 AI 怎么写 XML 的指令
const prompt = PromptTemplate.fromTemplate(
  "请根据用户输入，提取书籍信息并生成 XML。\n{format_instructions}\n用户输入: {query}"
);

// 4. 构建 LCEL 链 (在 TS 中同样使用 .pipe 语法，这对应 Python 的 |)
const XMLchain = prompt.pipe(model).pipe(XMLparser);

// 5. 调用
try {
  const result = await XMLchain.invoke({
    query: "《三体》的作者是刘慈欣，出版于2008年。",
    format_instructions: XMLparser.getFormatInstructions(),
  });
  console.log("解析后的对象:", JSON.stringify(result, null, 2));
} catch (error) {
  console.error("解析出错:", error);
}