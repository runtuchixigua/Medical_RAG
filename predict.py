from langchain import PromptTemplate  # 导入Langchain库中的PromptTemplate模块
from get_vector import *  # 导入自定义的获取向量的函数或类（具体内容不明确）
from model import ChatGLM_MD  # 导入自定义的ChatGLM_MD模型类（具体内容不明确）
from transformers import AutoModel, AutoTokenizer  # 导入Hugging Face的AutoModel和AutoTokenizer，用于加载模型和分词器

# 指定嵌入模型路径
EMBEDDING_MODEL = r'D:\AIproject\Pycharmproject\LLM\Custom_llm\moka-ai\m3e-base'
# 加载嵌入模型，使用HuggingFaceEmbeddings来生成嵌入向量
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# 加载FAISS数据库，FAISS是用于快速相似性搜索的库
db = FAISS.load_local('faiss', embeddings, allow_dangerous_deserialization=True)


# 获取与查询问题相关的内容
def get_related_content(related_docs):
    related_content = []  # 初始化一个空列表来存储相关内容
    for doc in related_docs:
        # 将文档中的内容按行处理，将两个换行符替换为一个换行符
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    # 将所有文档的相关内容合并为一个字符串，用换行符分隔
    return "\n".join(related_content)


# 定义用于生成问答的提示语（Prompt）
def define_prompt():
    question = '在浴室肾虚怎么办'  # 假设的用户问题
    # 使用FAISS数据库进行相似度搜索，找到与问题最相关的文档（k=1表示返回一个最相关文档）
    docs = db.similarity_search(question, k=1)
    # 提取相关文档的内容
    related_content = get_related_content(docs)

    # 定义一个模板，用于生成问答的提示语
    PROMPT_TEMPLATE = """
        基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
        已知内容:
        {context}
        问题:
        {question}"""

    # 创建PromptTemplate对象，并指定输入变量（'context'和'question'）
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=PROMPT_TEMPLATE,
    )

    # 使用上述模板格式化输入内容（将文档内容和问题填充到模板中）
    my_pmt = prompt.format(context=related_content,
                           question=question)

    return my_pmt  # 返回格式化后的提示语


# 定义QA流程函数
def qa():
    # 实例化自定义的ChatGLM_MD模型
    llm = ChatGLM_MD()
    # 加载指定路径的预训练模型
    llm.load_model(r'D:\modelfile\chatglm3-6b')
    # 生成提示语
    my_pmt = define_prompt()
    # 使用模型生成回答
    result = llm(my_pmt)
    return result  # 返回生成的回答


# 主函数，执行问答流程
if __name__ == '__main__':
    # 执行qa函数，得到结果
    result = qa()
    # 打印输出结果
    print(result)
