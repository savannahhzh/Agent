## 学习文档：
langchain: https://docs.langchain.com/oss/javascript/langchain/overview
博客： https://www.cuiliangblog.cn/detail/section/228761176


## Models
### model分类
LangChain中将大语言模型分为以下几种，我们主要使用的是对话模型：
1、非对话模型：输入字符串，输出字符串，不支持多轮对话上下文
2、对话模型：
3、嵌入模型


### 关键方法
1、invoke: 该模型以消息为输入，生成完整响应后输出消息；
2、stream: 调用模型，但实时流式传输输出；
3、batch: 批量向模型发送多个请求以实现更高效的处理
【与 invoke（） 不同，invoke（） 在模型生成完整响应后返回单个 AIMessage，stream（） 返回多个 AIMessageChunk 对象，每个对象包含输出文本的一部分。重要的是，流中的每个片段都设计成通过汇总汇集成完整消息：】

### Tools
1、定义的工具可供模型使用，必须使用bindTools绑定。

### Structured output 结构化输出
1、Zod  定义输出模式的首选方法；


## Messages
1、字段：
  role ——标识消息类型（例如系统 、 用户 ）
  content ——表示消息的实际内容（如文本、图片、音频、文档等）
  metadata - 可选字段，如响应信息、消息 ID 和令牌使用情况

2、Message Type
System message —— 告诉模型如何行为并为交互提供上下文；
Human message —— 表示用户输入和与模型的交互
AI message —— 模型生成的响应，包括文本内容、工具调用和元数据
Tool message —— 表示工具调用的输出
