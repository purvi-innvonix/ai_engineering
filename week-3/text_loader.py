from langchain_community.document_loaders import TextLoader

loader = TextLoader('makeup_overview.txt', encoding='utf-8')

docs = loader.load()
print(type(docs)) 