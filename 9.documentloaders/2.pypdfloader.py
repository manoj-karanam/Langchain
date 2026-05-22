from langchain_community.document_loaders import PyPDFLoader


loader= PyPDFLoader("../samplepdf.pdf")
docs=loader.load()

print(docs)