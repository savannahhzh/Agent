import "dotenv/config";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";

const Movie = z.object({
  title: z.string().describe("The title of the movie"),
  year: z.number().describe("The year the movie was released"),
  director: z.string().describe("The director of the movie"),
  rating: z.number().describe("The movie's rating out of 10")
});


const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
});

/**
 * 这个例子演示了如何使用模型的结构化输出功能来获取电影信息。模型将返回一个与上面定义的 Movie 模式相匹配的结构化响应。
 */
// const modelWithStructure = model.withStructuredOutput(Movie, {
//   method: "functionCalling",
//   includeRaw: true,
// });
// const response = await modelWithStructure.invoke("Provide details about the movie Inception");
// console.log(response);

/**
 *  多模态输出
 *  model.invoke() 与 contentBlocks
 *  1、contentBlocks 是什么
      contentBlocks 是 AIMessage 上的一个数组，每个元素代表一种内容块，结构如下：
      // 文本块
      { type: "text", text: "这是文字内容" }
      // 图片块
      { type: "image", source: { type: "base64", media_type: "image/png", data: "..." } }
      // 推理块（Claude 3.7 等思考模型特有）
      { type: "reasoning", thinking: "让我想一下..." }
      // 工具调用块
      { type: "tool_use", id: "...", name: "get_weather", input: { city: "上海" } }
       
  2、DeepSeek 不支持图片生成，deepseek-chat 是纯文本模型，调用会报错。图片生成需要 DALL·E（OpenAI）或 Stable Diffusion 等专用模型
  3、contentBlocks 是 Claude 系模型的概念，在 ChatAnthropic 上使用更自然。ChatOpenAI 的返回结构略有不同
*/
// const response = await model.invoke("Create a picture of a cat");
// console.log(response.contentBlocks);


/**
 * 这个例子演示了如何使用流式输出来获取模型的响应。
 * 模型会逐步返回响应内容，包括推理步骤和文本内容。
 */
const stream = await model.stream("Why do parrots have colorful feathers?");
for await (const chunk of stream) {
    const reasoningSteps = chunk.contentBlocks.filter(b => b.type === "reasoning");
    console.log(reasoningSteps.length > 0 ? reasoningSteps : chunk.text);
}