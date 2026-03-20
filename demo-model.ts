import "dotenv/config";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";

const movieSchema = z.object({
  title: z.string().describe("电影名称"),
  genre: z.enum(["Action", "Comedy", "Sci-Fi"]).describe("电影分类"),
  rating: z.number().min(1).max(10).describe("推荐指数")
});

const model = new ChatOpenAI({ 
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
}).withStructuredOutput(movieSchema, {
  method: "functionCalling",
  includeRaw: true // includeRaw: true 时返回 { raw: AIMessage, parsed: { title, genre, rating } }，结构化数据在 parsed 里
});

const result = await model.invoke("帮我评价一下《星际穿越》");
console.log(result.parsed.genre); // 自动获得类型提示，不再是 raw string！
console.log(result); 