from haystack import Document

from sqlitevec_haystack.document_store import SqliteVecDocumentStore


document_store = SqliteVecDocumentStore(
    connection_string="integrations/sqlitevec-haystack/test.db", embedding_dimension=3, recreate_table=True
)
document_store.connection


documents = [
    Document(id="123", content="test", embedding=[0.3, 0.1, 10.0], meta={"a": 1}),
    Document(id="321", content="test2", embedding=[0.1, 0.3, 1.0], meta={"a": 2})
]

document_store.write_documents(documents)
filters = {
    "operator": "AND",
    "conditions":[
        {"field": "id", "operator": "==", "value": "123"},
        # {"field": "content", "operator": "==", "value": "test"}
    ]
}

document_store.delete_documents(document_ids=["321", "123"])
retrieved_documents = document_store.filter_documents(filters={})
# assert all(x in documents for x in retrieved_documents)

n_documents = document_store.count_documents()
assert n_documents == 2 
