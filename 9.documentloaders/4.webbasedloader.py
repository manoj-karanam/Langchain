from langchain_community.document_loaders import WebBaseLoader



url="https://techlekh.com/dongfeng-nammi-vigo-price-nepal/"

loader=WebBaseLoader(url)

docs=loader.load()

print(docs)