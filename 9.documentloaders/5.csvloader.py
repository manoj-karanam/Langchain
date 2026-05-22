from langchain_community.document_loaders import CSVLoader

# data.csv does not exist
loader=CSVLoader("data.csv")
data=loader.load()

print(data)
