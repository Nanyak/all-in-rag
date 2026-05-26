# Section 3 Milvus index construction

In graph RAG systems, index construction is a key link between graph data and vector retrieval. This section describes how to convert graph data into searchable vector indices.

In Chapter 3, we have introduced the basic concepts, deployment methods and basic operations of Milvus in detail. This section will build on this foundation and conduct in-depth applications specifically for graph RAG scenarios. If you are not familiar with Milvus, it is recommended to read [Milvus introduction and multi-modal retrieval practice] (https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter3/09_milvus.md) first.

> [Complete code of this section](https://github.com/datawhalechina/all-in-rag/blob/main/code/C9/rag_modules/milvus_index_construction.py)

## 1. Overview of index construction

### 1.1 Index construction process

The index construction of graph RAG requires converting the structured documents built from the graph database into vector representations and storing them in the vector database:

```mermaid
flowchart LR
    A[图数据库] --> B[文档构建]
    B --> C[文档分块]
    C --> D[向量化]
    D --> E[Milvus索引]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
```

### 1.2 Core components

- **Document Builder**: Build structured documents from graph data
- **Chunking Processor**: Intelligent chunking strategy
- **Vectorization model**: Text steering vector
- **Milvus Index**: high-performance vector storage and retrieval

## 2. Implementation of Milvus index construction

### 2.1 Index builder core architecture

```python
class MilvusIndexConstructionModule:
    """Milvus索引构建模块 - 负责向量化和Milvus索引构建"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()
```

**Code Interpretation**:
- **Modular design**: Encapsulate Milvus operations into independent modules for easy reuse and maintenance
- **Configuration Flexibility**: Supports custom Milvus connection parameters and embedded models
- **Chinese Optimization**:`BAAI/bge-small-zh-v1.5`is used by default, specifically optimized for Chinese text
- **Lazy initialization**: Set the connection in the constructor to avoid blocking at startup

### 2.2 Vectorization processing

```python
def _vectorize_documents(self, documents: List[Document]) -> Tuple[List[List[float]], List[Dict]]:
    """文档向量化处理"""
    vectors = []
    metadatas = []
    
    for i, doc in enumerate(documents):
        try:
            # 向量化文档内容
            vector = self.embedding_model.embed_query(doc.page_content)
            vectors.append(vector)
            
            # 准备元数据
            metadata = {
                "id": i,
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
                # ... 其他元数据
            }
            metadatas.append(metadata)
            
        except Exception as e:
            logger.error(f"文档 {i} 向量化失败: {e}")
            continue
    
    return vectors, metadatas
```

### 2.3 Figure RAG special collection Schema design

```python
def _create_collection_schema(self):
    """创建集合schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
        FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="difficulty", dtype=DataType.INT64),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="中式烹饪知识图谱向量集合"
    )
    return schema
```

**Schema design highlights**:
- **Graph data specialization**: Field structure specially designed for cooking knowledge graphs
- **Rich metadata**: Contains map-specific information such as recipe name, node type, cuisine, difficulty, etc.
- **Length Optimization**: Set reasonable field length limits based on actual data characteristics
- **Search Friendly**: All key fields are available for filtering and search criteria

## 3. Index optimization strategy

### 3.1 Batch insertion optimization

```python
def _batch_insert(self, vectors: List[List[float]], metadatas: List[Dict]):
    """批量插入优化"""
    batch_size = self.config.batch_size
    collection_name = self.config.milvus_collection_name
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        # 准备插入数据
        insert_data = [
            [meta["id"] for meta in batch_metadatas],           # id
            batch_vectors,                                       # vector
            [meta["content"] for meta in batch_metadatas],      # content
            [meta["source"] for meta in batch_metadatas],       # source
            [meta["chunk_id"] for meta in batch_metadatas],     # chunk_id
            [meta["parent_id"] for meta in batch_metadatas],    # parent_id
        ]
        
        # 执行插入
        self.milvus_client.insert(collection_name, insert_data)
        logger.info(f"批次 {i//batch_size + 1} 插入完成，数量: {len(batch_vectors)}")
```

### 3.2 Index creation

```python
def _create_index(self):
    """创建向量索引"""
    collection_name = self.config.milvus_collection_name
    
    # 索引参数
    index_params = {
        "metric_type": "COSINE",    # 余弦相似度
        "index_type": "IVF_FLAT",   # 索引类型
        "params": {"nlist": 1024}   # 索引参数
    }
    
    # 创建索引
    self.milvus_client.create_index(
        collection_name=collection_name,
        field_name="vector",
        index_params=index_params
    )
    
    # 加载集合到内存
    self.milvus_client.load_collection(collection_name)
    
    logger.info("向量索引创建完成")
```

## 4. Index construction process

### 4.1 Core vector construction process

```python
def build_vector_index(self, chunks: List[Document]) -> bool:
    """构建向量索引"""
    logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

    try:
        # 1. 创建集合（如果schema不兼容则强制重新创建）
        if not self.create_collection(force_recreate=True):
            return False

        # 2. 准备数据
        logger.info("正在生成向量embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # 3. 准备插入数据
        entities = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            entity = {
                "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                "vector": vector,
                "text": self._safe_truncate(chunk.page_content, 15000),
                "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                # ... 更多字段
            }
            entities.append(entity)

        # 4. 批量插入数据
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
```

**Interpretation of key technical points**:

1. **Forced reconstruction strategy**:`force_recreate=True`ensures Schema consistency and avoids field mismatch errors

2. **Batch vectorization**: Process the vectorization of all documents at one time to improve efficiency
   ```python
   texts = [chunk.page_content for chunk in chunks]
   vectors = self.embeddings.embed_documents(texts)  # 批量处理
   ```

3. **Safety truncation mechanism**:`_safe_truncate`method prevents field length from exceeding the limit
   ```python
   def _safe_truncate(self, text: str, max_length: int) -> str:
       if text is None:
           return ""
       return str(text)[:max_length]
   ```

4. **Graph data metadata retention**: Completely retain the structured information in the graph to support subsequent compound searches

### 4.2 Index verification

```python
def verify_index(self) -> bool:
    """验证索引构建结果"""
    try:
        collection_name = self.config.milvus_collection_name
        
        # 检查集合状态
        collection_info = self.milvus_client.describe_collection(collection_name)
        logger.info(f"集合信息: {collection_info}")
        
        # 检查数据量
        count = self.milvus_client.query(
            collection_name=collection_name,
            expr="",
            output_fields=["count(*)"]
        )
        logger.info(f"索引中文档数量: {count}")
        
        # 简单检索测试
        test_results = self.milvus_client.search(
            collection_name=collection_name,
            data=[[0.1] * self.config.embedding_dim],  # 测试向量
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1
        )
        
        logger.info("索引验证通过")
        return True

    except Exception as e:
        logger.error(f"索引验证失败: {e}")
        return False
```

## 5. Why switch from FAISS to Milvus?

In Chapter 8, FAISS is used as the vector storage scheme. While FAISS excels in research and prototype development, Milvus offers additional advantages in production environments and complex application scenarios:

**Limitations of FAISS**:
- **Pure library mode**: FAISS is a vector search library and lacks the full functionality of a database
- **No persistence**: Requires manual management of data persistence and backup
- **Single machine limitation**: It is difficult to achieve distributed deployment and horizontal expansion
- **Limited Metadata Support**: Unable to efficiently store and query complex structured metadata
- **Concurrency Performance**: Performance is limited in high concurrency scenarios

**Milvus Advantages**:
- **Complete database functions**: Provide CRUD operations, transaction support, and data consistency guarantee
- **Cloud native architecture**: supports distributed deployment, automatic expansion and contraction, and high availability
- **Rich metadata support**: Supports complex Schema design, suitable for multi-dimensional data of graph RAG
- **Production-level features**: enterprise-level functions such as monitoring, logging, backup and recovery, etc.
