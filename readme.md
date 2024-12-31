---

# Medical_RAG 项目

本项目旨在构建一个医疗文本分析与问答系统，通过使用自然语言处理技术，将医疗相关的长文本分割、嵌入向量，并存储于FAISS数据库中，以便快速检索和生成基于上下文的回答。



## 文件说明

- `get_vector.py`：用于加载文本文件，将其分割为小块，并使用HuggingFace嵌入模型将文本转换为向量，存储于FAISS数据库中。
- `model.py`：定义了一个基于Langchain的LLM类，用于加载预训练的模型和分词器，并生成回答。
- `predict.py`：定义了问答流程，包括从FAISS数据库中检索相关内容，并使用`ChatGLM_MD`模型生成回答。

## 依赖项

本项目使用以下Python库和框架：

- `langchain`：一个用于构建语言模型应用的框架。
- `transformers`：Hugging Face的Transformers库，用于加载和使用预训练的模型。
- `torch`：PyTorch深度学习框架，用于模型推理。
- `langchain_community`：Langchain社区提供的扩展，用于增强LLM功能。

你可以通过以下命令安装项目的依赖项：

```bash
pip install -r requirements.txt
```


## 使用方法

### 1. 克隆项目

首先，克隆项目到本地：

```bash
git clone https://github.com/your-repo/chatglm-md.git
cd chatglm-md
```

### 2. 安装依赖

确保已经安装了Python并激活了虚拟环境，然后安装项目的依赖项：

```bash
pip install -r requirements.txt
```

### 3. 加载和运行模型

在`model.py`中，`Medical_RAG`类定义了加载和使用`ChatGLM`模型的功能。你可以通过以下步骤来加载模型并与之进行对话：

#### 代码示例：

```python
from model import Medical_RAG

# 创建Medical_RAG模型的实例
model = Medical_RAG()

# 加载预训练的ChatGLM模型，模型路径根据实际情况设置
model.load_model('D:/modelfile/chatglm-6b')

# 使用加载的模型进行对话
response = model('您好！')

# 输出模型的回答
print(response)
```

- `load_model`方法加载预训练的`ChatGLM`模型。你需要提供模型所在的路径，例如`'D:/modelfile/chatglm-6b'`。
- 调用模型时，输入一个问题或对话内容，模型会返回一个生成的回答。

### 4. 配置文件

确保在本地设置正确的模型路径，路径应包含预训练的`ChatGLM`模型和分词器。如果没有该模型，你可以在Hugging Face模型库中下载。

## 模型参数

`Medical_RAG`类中的一些关键参数可以在初始化时配置，具体包括：

- `max_token`: 控制生成文本的最大token数（默认为4096）。
- `temperature`: 控制生成文本的随机性（默认为0.8）。
- `top_p`: 控制生成文本的多样性（默认为0.9）。

### 主要方法

#### `load_model(model_path)`

- **参数**：`model_path` (str) - 模型文件夹路径，包含`ChatGLM`模型和分词器。
- **功能**：加载指定路径的预训练模型和分词器。

#### `__call__(prompt)`

- **参数**：`prompt` (str) - 用户输入的对话内容。
- **返回**：返回模型生成的文本响应。
- **功能**：根据给定的提示生成回答，并返回结果。

### 5. 对话历史

`Medical_RAG`类维护了`history`（历史对话记录）以便于在多轮对话中传递上下文。每次调用模型生成回答时，历史记录会被更新，并且生成的文本会加入到历史中。

## 常见问题 (FAQ)

### Q1: 如何改变生成的文本长度？

可以通过设置`max_token`参数来控制文本生成的最大长度。例如，将其设置为1024：

```python
model.max_token = 1024
```

### Q2: 如何控制生成的文本的随机性？

可以通过设置`temperature`参数来控制生成文本的随机性。较高的`temperature`会增加生成文本的多样性，较低的`temperature`
会使输出更加确定性。例如：

```python
model.temperature = 0.7
```



