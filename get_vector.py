from langchain_community.document_loaders import TextLoader  # 从Langchain社区加载文档加载器，用于加载文本文件
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 从Langchain加载文本切分器，用于将长文本分割为较小的块
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # 从Langchain加载HuggingFace嵌入模型，用于将文本转换为向量
from langchain.vectorstores import FAISS  # 导入FAISS库，用于存储和检索向量


def main():
    # 加载文本文件，指定路径和编码
    textloader = TextLoader(r'D:\AIproject\Pycharmproject\LLM\medical_rag\data\output.txt', encoding='utf-8')

    # 使用textloader加载文档
    document = textloader.load()

    # 使用递归字符分割器将文档切分为小块，chunk_size指定每块的最大字符数，chunk_overlap指定块之间的重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)

    # 打印切分后的文档块数
    print(len(split_docs))

    # 加载HuggingFace嵌入模型，模型路径指定了自定义的嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=r'D:\AIproject\Pycharmproject\LLM\Custom_llm\moka-ai\m3e-base')

    # 使用FAISS库从分割后的文档创建一个向量数据库，并将文档嵌入为向量
    db = FAISS.from_documents(split_docs, embeddings)

    # 将生成的向量数据库保存到本地
    db.save_local("faiss/")

    # 返回分割后的文档（此返回值在当前代码中未使用）
    return split_docs


# 程序入口，调用main函数
if __name__ == "__main__":
    main()
