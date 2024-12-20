from haystack import Document

from aai_assistant.sqlite_vec.document_store import SqliteVecDocumentStore


document_store = SqliteVecDocumentStore(
    connection_string="test.db", embedding_dimension=3, recreate_table=True
)
document_store.connection


documents = [
    Document(id="123", content="test", embedding=[0.3, 0.1, 10.0], meta={"a": 1}),
    Document(id="321", content="test2", embedding=[0.1, 0.3, 1.0], meta={"a": 2})
]

document_store.write_documents(documents)
retrieved_documents = document_store.filter_documents()

# assert all(x in documents for x in retrieved_documents)

n_documents = document_store.count_documents()
assert n_documents == 2 
