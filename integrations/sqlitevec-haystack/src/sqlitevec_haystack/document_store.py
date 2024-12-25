import logging
import sqlite3
import json
from typing import Any, Dict, List, Literal, Tuple, Optional
from struct import unpack

import sqlite_vec
from sqlite_vec import serialize_float32
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from .filters import _convert_filters_to_where_clause_and_params

logger = logging.getLogger(__name__)

# TODO: create a main table for sqlite, and two virtual tables for vector and fts
CREATE_MAIN_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS {table_name} (
id VARCHAR(128) PRIMARY KEY,
vec_id INTEGER UNIQUE,
content TEXT,
dataframe JSONB,
blob_data BYTE,
blob_meta JSONB,
blob_mime_type VARCHAR(255),
meta JSONB)
"""

CREATE_VEC_TABLE_STATEMENT = """
CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_vec USING vec0(
vec_id INTEGER PRIMARY KEY AUTOINCREMENT,
embedding FLOAT[{embedding_dimension}])"""

INSERT_MAIN_TABLE_STATEMENT = """
INSERT INTO {table_name}
(id, vec_id, content, dataframe, blob_data, blob_meta, blob_mime_type, meta)
VALUES (:id, :vec_id, :content, :dataframe, :blob_data, :blob_meta, :blob_mime_type, :meta)
"""

INSERT_VEC_TABLE_STATEMENT = """
INSERT INTO {table_name}_vec(vec_id, embedding)
VALUES (:vec_id, :embedding)
"""

UPDATE_MAIN_TABLE_STATEMENT = """
ON CONFLICT (id) DO UPDATE SET
vec_id = EXCLUDED.vec_id,
content = EXCLUDED.content,
dataframe = EXCLUDED.dataframe,
blob_data = EXCLUDED.blob_data,
blob_meta = EXCLUDED.blob_meta,
blob_mime_type = EXCLUDED.blob_mime_type,
meta = EXCLUDED.meta
"""

UPDATE_VEC_TABLE_STATEMENT = """
ON CONFLICT (vec_id) DO UPDATE SET
embedding = EXCLUDED.embedding
"""

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "inner_product", "l2_distance"]

VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_similarity": "vector_cosine_ops",
    "inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]



def deserialize_float32(raw_bytes: bytes, embedding_dim: int) -> List[float]:
    """Deserializes raw bytes into a list of floats"""
    return unpack(f"{embedding_dim}f", raw_bytes)

class SqliteVecDocumentStore:
    """
    A Document Store using sqlite with the [sqlite-vec extension]() installed.
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
        table_name: str = "haystack_documents",
        language: str = "english",
        embedding_dimension: int = 768,
        vector_function: Literal[
            "cosine_similarity", "inner_product", "l2_distance"
        ] = "cosine_similarity",
        recreate_table: bool = False,
        search_strategy: Literal[
            "exact_nearest_neighbor", "hnsw"
        ] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: Optional[Dict[str, int]] = None,
        hnsw_index_name: str = "haystack_hnsw_index",
        hnsw_ef_search: Optional[int] = None,
        keyword_index_name: str = "haystack_keyword_index",
    ):
        """
        Creates a new PgvectorDocumentStore instance.
        It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param connection_string: The connection string to use to connect to the PostgreSQL database, defined as an
            environment variable. It can be provided in either URI format
            e.g.: `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"`, or keyword/value format
            e.g.: `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
            See [PostgreSQL Documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
            for more details.
        :param table_name: The name of the table to use to store Haystack documents.
        :param language: The language to be used to parse query and document content in keyword retrieval.
            To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
            `SELECT cfgname FROM pg_ts_config;`.
            More information can be found in this [StackOverflow answer](https://stackoverflow.com/a/39752553).
        :param embedding_dimension: The dimension of the embedding.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            `"cosine_similarity"` and `"inner_product"` are similarity functions and
            higher scores indicate greater similarity between the documents.
            `"l2_distance"` returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param recreate_table: Whether to recreate the table if it already exists.
        :param search_strategy: The search strategy to use when searching for similar embeddings.
            `"exact_nearest_neighbor"` provides perfect recall but can be slow for large numbers of documents.
            `"hnsw"` is an approximate nearest neighbor search strategy,
            which trades off some accuracy for speed; it is recommended for large numbers of documents.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param hnsw_recreate_index_if_exists: Whether to recreate the HNSW index if it already exists.
            Only used if search_strategy is set to `"hnsw"`.
        :param hnsw_index_creation_kwargs: Additional keyword arguments to pass to the HNSW index creation.
            Only used if search_strategy is set to `"hnsw"`. You can find the list of valid arguments in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
        :param hnsw_index_name: Index name for the HNSW index.
        :param hnsw_ef_search: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
            `"hnsw"`. You can find more information about this parameter in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
        :param keyword_index_name: Index name for the Keyword index.
        """

        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)
        self.vector_function = vector_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_index_name = hnsw_index_name
        self.hnsw_ef_search = hnsw_ef_search
        self.keyword_index_name = keyword_index_name
        self.language = language
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self):
        if self._connection is None:
            self._connection = self._create_connection()

        return self._connection

    def _create_connection(self):
        # We pass sqlite3.PARSE_DECLTYPES to detect_types
        # in order for the connection to use the registered converter
        connection = sqlite3.connect(self.connection_string, detect_types=sqlite3.PARSE_DECLTYPES)
        connection.set_trace_callback(print)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        # connection.load_extension("fts5")
        connection.enable_load_extension(False)
        
        # Handle conversion from dict -> json string and back
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.register_adapter
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.register_converter
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_converter("jsonb", json.loads)

        with connection as cursor:
            # Init schema
            if self.recreate_table:
                self.delete_table(cursor)
            self._create_main_table_if_not_exists(cursor)
            self._create_vec_table_if_not_exists(cursor)
            # self._create_keyword_index_if_not_exists() # TODO:full-text search.

        return connection

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
            search_strategy=self.search_strategy,
            hnsw_recreate_index_if_exists=self.hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=self.hnsw_index_creation_kwargs,
            hnsw_index_name=self.hnsw_index_name,
            hnsw_ef_search=self.hnsw_ef_search,
            keyword_index_name=self.keyword_index_name,
            language=self.language,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SqliteVecDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["connection_string"])
        return default_from_dict(cls, data)

    def _execute_sql(
        self,
        sql_query: str,
        cursor: sqlite3.Cursor,
        params: Optional[tuple] = None,
        error_msg: str = "",
    ):
        """
        Internal method to execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param cursor: The cursor to use to execute the SQL query.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        """

        params = params or ()

        sql_query_str = (
            sql_query.as_string(cursor) if not isinstance(sql_query, str) else sql_query
        )
        logger.debug("SQL query: %s\nParameters: %s", sql_query_str, params)

        try:
            result = cursor.execute(sql_query, params)
        except Exception as e:
            detailed_error_msg = f"{error_msg}.\nYou can find the SQL query and the parameters in the debug logs."
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    def _create_main_table_if_not_exists(self, cursor: sqlite3.Cursor):
        """
        Creates the main table to store Haystack documents if it doesn't exist yet.
        """

        create_sql = CREATE_MAIN_TABLE_STATEMENT.format(
            table_name=self.table_name
        )

        self._execute_sql(
            create_sql, cursor=cursor, error_msg=f"Could not create table in {self.__class__.__name__}"
        )

    def _create_vec_table_if_not_exists(self, cursor: sqlite3.Cursor):
        """
        Creates the vec table to store Haystack documents' embeddings if it doesn't exist yet.
        """

        create_sql = CREATE_VEC_TABLE_STATEMENT.format(
            table_name=self.table_name, embedding_dimension=self.embedding_dimension
        )

        self._execute_sql(
            create_sql, cursor=cursor, error_msg=f"Could not create table in {self.__class__.__name__}"
        )

    def delete_table(self, cursor: sqlite3.Cursor):
        """
        Deletes the table used to store Haystack documents.
        The name of the table (`table_name`) is defined when initializing the `PgvectorDocumentStore`.
        """

        delete_sql = f"DROP TABLE IF EXISTS {self.table_name};"
        delete_vec_sql = f"DROP TABLE IF EXISTS {self.table_name}_vec;"

        self._execute_sql(
            delete_sql,
            cursor=cursor,
            error_msg=f"Could not delete table {self.table_name} in SqliteVecDocumentStore",
        )
        self._execute_sql(
            delete_vec_sql,
            cursor=cursor,
            error_msg=f"Could not delete vector table {self.table_name}_vec in SqliteVecDocumentStore",
        )
        
    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """

        sql_count = "SELECT COUNT(*) FROM {table_name}".format(
            table_name=self.table_name
        )

        with self._connection as cursor:
            count = self._execute_sql(
                sql_count, cursor=cursor, error_msg="Could not count documents in PgvectorDocumentStore"
            ).fetchone()[0]
        return count

    def filter_documents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises TypeError: If `filters` is not a dictionary.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            if not isinstance(filters, dict):
                msg = "Filters must be a dictionary"
                raise TypeError(msg)
            if "operator" not in filters and "conditions" not in filters:
                msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
                raise ValueError(msg)

        sql_filter = (
            "SELECT * FROM {table_name} "
            "LEFT JOIN {table_name}_vec " 
            "ON {table_name}.vec_id = {table_name}_vec.vec_id "
        ).format(
            table_name=self.table_name
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters
            )
            sql_filter += sql_where_clause

        with self.connection as connection:
            cursor = connection.cursor()
            result = self._execute_sql(
                sql_filter,
                cursor=cursor,
                params=params,
                error_msg=f"Could not filter documents from {self.__class__.__name__}",
            )
            
            records = result.fetchall()
            docs = self._from_sqlite_to_haystack_documents(records)
            return docs

    def write_documents(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :returns: The number of documents written to the document store.
        """

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = (
                    "param 'documents' must contain a list of objects of type Document"
                )
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents, db_vectors = self._from_haystack_to_sqlite_documents(documents)

        sql_main_table_insert = INSERT_MAIN_TABLE_STATEMENT.format(
            table_name=(self.table_name)
        )
        sql_vec_table_insert = INSERT_VEC_TABLE_STATEMENT.format(
            table_name=self.table_name
        )

        if policy == DuplicatePolicy.OVERWRITE:
            sql_main_table_insert += UPDATE_MAIN_TABLE_STATEMENT
            sql_vec_table_insert += UPDATE_VEC_TABLE_STATEMENT
        elif policy == DuplicatePolicy.SKIP:
            sql_main_table_insert += ("ON CONFLICT DO NOTHING")
            sql_vec_table_insert += ("ON CONFLICT DO NOTHING")

        sql_main_table_query_str = (
            sql_main_table_insert.as_string(self.cursor)
            if not isinstance(sql_main_table_insert, str)
            else sql_main_table_insert
        )
        
        sql_vec_table_query_str = (
            sql_vec_table_insert.as_string(self.cursor)
            if not isinstance(sql_vec_table_insert, str)
            else sql_vec_table_insert
        )

        written_docs = 0
        for db_document, db_vector in zip(db_documents, db_vectors):
            logger.debug(" query: %s\nParameters: %s", sql_vec_table_query_str, db_vector)

            with self._connection as connection:
                cursor = connection.cursor()
                try:
                    db_vector["vec_id"] = None
                    db_vector["embedding"] = serialize_float32(db_vector["embedding"])
                    cursor.execute(sql_vec_table_insert, db_vector)
                except Exception as ie:
                    raise DuplicateDocumentError from ie
                except Exception as e:
                    error_msg = (
                        f"Could not write documents to {self.__class__.__name__}. \n"
                        "You can find the SQL query and the parameters in the debug logs."
                    )
                    raise DocumentStoreError(error_msg) from e

                vec_id = cursor.lastrowid
                db_document["vec_id"] = vec_id
                # db_document["meta"] = json.dumps(db_document["meta"])
                
                logger.debug(" query: %s\nParameters: %s", sql_main_table_query_str, db_document)
            
                try:
                    cursor.execute(sql_main_table_insert, db_document)
                except Exception as ie:
                    raise DuplicateDocumentError from ie
                except Exception as e:
                    error_msg = (
                        f"Could not write documents to {self.__class__.__name__}. \n"
                        "You can find the SQL query and the parameters in the debug logs."
                    )
                    raise DocumentStoreError(error_msg) from e

            written_docs += 1
        
        return written_docs

    @staticmethod
    def _from_haystack_to_sqlite_documents(
        documents: List[Document],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Internal method to convert a list of Haystack Documents to a list of dictionaries that can be used to insert
        documents into the PgvectorDocumentStore.
        """

        db_vectors = []
        db_documents = []
        
        for document in documents:
            document_dict = document.to_dict(flatten=False)
            db_document = {
                k: v
                for k, v in document_dict.items()
                if k not in ["score", "blob", "embedding"]
            }
            db_vector = {"embedding": document_dict["embedding"]}
            blob = document.blob
            db_document["blob_data"] = blob.data if blob else None
            db_document["blob_meta"] = blob.meta if blob and blob.meta else None
            db_document["blob_mime_type"] = (
                blob.mime_type if blob and blob.mime_type else None
            )

            db_document["dataframe"] = (
                db_document["dataframe"] if db_document["dataframe"] else None
            )
            db_document["meta"] = db_document["meta"]

            if "sparse_embedding" in db_document:
                sparse_embedding = db_document.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document %s has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in Sqlite is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        db_document["id"],
                    )

            db_documents.append(db_document)
            db_vectors.append(db_vector)
        return db_documents, db_vectors
    
    def _from_sqlite_to_haystack_documents(
        self, documents: List[sqlite3.Row],
    ) -> List[Document]:
        """
        Internal method to convert a list of dictionaries from pgvector to a list of Haystack Documents.
        """

        haystack_documents = []
        for document in documents:
            haystack_dict = dict(document)
            blob_data = haystack_dict.pop("blob_data")
            blob_meta = haystack_dict.pop("blob_meta")
            blob_mime_type = haystack_dict.pop("blob_mime_type")
            vec_id = haystack_dict.pop("vec_id")

            # postgresql returns the embedding as a string
            # so we need to convert it to a list of floats
            if haystack_dict.get("embedding") is not None:
                haystack_dict["embedding"] = deserialize_float32(haystack_dict["embedding"], self.embedding_dimension)

            print(f"{haystack_dict}")
            haystack_document = Document.from_dict(haystack_dict)

            if blob_data:
                blob = ByteStream(
                    data=blob_data, meta=blob_meta, mime_type=blob_mime_type
                )
                haystack_document.blob = blob

            haystack_documents.append(haystack_document)

        return haystack_documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.
        Also removes the associated rows in the vector table, matching by `vec_id`.

        :param document_ids: the document IDs to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join("?" for _ in document_ids)

        # Find the vec_ids for the documents we want to delete
        select_vec_ids_sql = f"""
        SELECT vec_id FROM {self.table_name}
        WHERE id IN ({document_ids_str})
        """

        with self.connection as connection:
            cursor = connection.cursor()

            # Fetch all vec_ids for matching documents
            cursor.execute(select_vec_ids_sql, document_ids)
            found_vec_ids = [row["vec_id"] for row in cursor.fetchall() if row["vec_id"]]

            # Delete from the main table
            delete_main_sql = f"""
            DELETE FROM {self.table_name}
            WHERE id IN ({document_ids_str})
            """
            cursor.execute(delete_main_sql, document_ids)

            # Delete the corresponding rows in the vector table
            if found_vec_ids:
                document_vec_ids_str = ", ".join("?" for _ in found_vec_ids)
                delete_vec_sql = f"""
                DELETE FROM {self.table_name}_vec
                WHERE vec_id IN ({document_vec_ids_str})
                """
                cursor.execute(delete_vec_sql, found_vec_ids)

    def _keyword_retrieval(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query using a full-text search.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorKeywordRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query`
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        sql_select = SQL(KEYWORD_QUERY).format(
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
            query=SQLLiteral(query),
        )

        where_params = ()
        sql_where_clause = SQL("")
        if filters:
            sql_where_clause, where_params = (
                _convert_filters_to_where_clause_and_params(
                    filters=filters, operator="AND"
                )
            )

        sql_sort = SQL(" ORDER BY score DESC LIMIT {top_k}").format(
            top_k=SQLLiteral(top_k)
        )

        sql_query = sql_select + sql_where_clause + sql_sort

        result = self._execute_sql(
            sql_query,
            (query, *where_params),
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
            cursor=self.dict_cursor,
        )

        records = result.fetchall()
        docs = self._from_pg_to_haystack_documents(records)
        return docs

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[
            Literal["cosine_similarity", "inner_product", "l2_distance"]
        ] = None,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorEmbeddingRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        if len(query_embedding) != self.embedding_dimension:
            msg = (
                f"query_embedding dimension ({len(query_embedding)}) does not match PgvectorDocumentStore "
                f"embedding dimension ({self.embedding_dimension})."
            )
            raise ValueError(msg)

        vector_function = vector_function or self.vector_function
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        # the vector must be a string with this format: "'[3,1,2]'"
        query_embedding_for_postgres = (
            f"'[{','.join(str(el) for el in query_embedding)}]'"
        )

        # to compute the scores, we use the approach described in pgvector README:
        # https://github.com/pgvector/pgvector?tab=readme-ov-file#distances
        # cosine_similarity and inner_product are modified from the result of the operator
        if vector_function == "cosine_similarity":
            score_definition = (
                f"1 - (embedding <=> {query_embedding_for_postgres}) AS score"
            )
        elif vector_function == "inner_product":
            score_definition = (
                f"(embedding <#> {query_embedding_for_postgres}) * -1 AS score"
            )
        elif vector_function == "l2_distance":
            score_definition = f"embedding <-> {query_embedding_for_postgres} AS score"

        sql_select = SQL("SELECT *, {score} FROM {table_name}").format(
            table_name=Identifier(self.table_name),
            score=SQL(score_definition),
        )

        sql_where_clause = SQL("")
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters
            )

        # we always want to return the most similar documents first
        # so when using l2_distance, the sort order must be ASC
        sort_order = "ASC" if vector_function == "l2_distance" else "DESC"

        sql_sort = SQL(" ORDER BY score {sort_order} LIMIT {top_k}").format(
            top_k=SQLLiteral(top_k),
            sort_order=SQL(sort_order),
        )

        sql_query = sql_select + sql_where_clause + sql_sort

        result = self._execute_sql(
            sql_query,
            params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
            cursor=self.dict_cursor,
        )

        records = result.fetchall()
        docs = self._from_sqlite_to_haystack_documents(records)
        return docs
