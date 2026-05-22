from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


# these pdf's have sme issues but the code is right
loader= DirectoryLoader(path="RAG", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)

docs=loader.load()

for doc in docs:
    print(doc.metadata)