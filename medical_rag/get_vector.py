from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS # 向量数据库







def main():
    textloader = TextLoader(r'D:\AIproject\Pycharmproject\LLM\medical_rag\data\output.txt',encoding='utf-8')

    document = textloader.load()
    # print(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)

    # print(split_docs)
    print(len(split_docs))




    embeddings = HuggingFaceEmbeddings(model_name=r'D:\AIproject\Pycharmproject\LLM\Custom_llm\moka-ai\m3e-base')
    # print(embeddings)
    db = FAISS.from_documents(split_docs,embeddings)
    db.save_local("faiss/")



    return split_docs
if __name__ == "__main__":
    main()

