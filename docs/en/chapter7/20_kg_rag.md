# Section 1 RAG based on knowledge graph

Although the traditional RAG framework effectively alleviates the knowledge obsolescence and hallucination problems of large language models, it still has obvious limitations in processing complex queries. Methods that rely on unstructured text vector retrieval are often difficult to capture the deep relationships between entities, leading to inaccurate context retrieval, information fragmentation, and even inducing "hallucinations" in the model. In order to break through these bottlenecks, GraphRAG, a new paradigm that integrates structured knowledge graphs into the RAG process, was born. By taking advantage of the explicit semantic relationships and graph structure of the knowledge graph, GraphRAG can provide more accurate context retrieval and stronger reasoning capabilities, and performs particularly well in multi-hop queries and scenarios with high factual requirements.

## 1. The evolution from traditional RAG to knowledge graph enhanced RAG

<div align="center">
   <img src="./images/7_1_1.svg" alt="GraphRAG" width="800">
</div>

### 1.1 Inherent limitations of traditional RAG framework

Although traditional RAG solves the knowledge update problem of LLM to a certain extent through the two-stage "retrieval-generation" process, its core mechanism based on unstructured text vector retrieval still has several key limitations.

**Lack of relationship understanding**: Vector-based retrieval mainly focuses on semantic similarity, and it is difficult to capture and utilize complex and implicit relationships between entities. When the query involves multi-entity association or causal reasoning, the retrieved text blocks may lack logical connections, making it difficult for LLM to perform effective synthesis and reasoning.

**Fragmentation of context**: Text is split into independent chunks for indexing, which destroys the structure and contextual continuity of the original text. For problems that require information integration across multiple documents or long-distance texts, traditional RAG is often inadequate.

**Retrieval noise and hallucination risk**: The retrieval process may return irrelevant or only partially relevant noise information, which will interfere with LLM's judgment and even induce it to produce "hallucination" content that is inconsistent with the facts. The introduction of knowledge graphs has been proven to effectively reduce model illusion and provide higher quality context.

**Limited reasoning capability**: The reasoning capability of traditional RAG is limited by the retrieved linear text content, making it difficult to support multi-hop reasoning that requires structured knowledge for navigation.

**Weak cross-document connection ability**: Even with larger tiles and sliding window overlaps, entity co-occurrence and implicit references across documents/cross-sections are still difficult to explicitly model, resulting in insufficient recall of questions that require "connecting edges" to answer (such as tracing along cause and effect, time, or supply chain relationships).

**Entity ambiguity and alias issues**: It is difficult to stably distinguish entities with the same name (such as different cities/companies/person names) by relying only on vector similarity, or to unify aliases, abbreviations and multi-language variants, which can easily lead to misalignment and semantic drift.

**Insufficient timeliness and version consistency**: Text blocks often lack calculable time attributes and validity boundaries, making it difficult to get consistent answers to timing-sensitive questions (such as "Is a company still a subsidiary in a certain year?").

A typical example:

- Question: "For the company that acquired Company A in 2019, who is the main investment target of its parent company in 2021?"
- The traditional approach requires retrieving multiple paragraphs of text and splicing and inferring on its own during the generation phase; if any paragraph is missing or ambiguous, the overall reasoning chain will be broken.

### 1.2 The core advantages of knowledge graph empowering RAG

Knowledge graph explicitly organizes discrete knowledge into an interconnected semantic network through the network structure of nodes (entities) and edges (relationships), providing a powerful solution to overcome the limitations of traditional RAG.

**Structured Semantic Expression**: The knowledge graph directly encodes explicit semantic relationships between entities (such as "Company A - Acquisition - Company B") in the form of a graph, avoiding LLM's implicit and potentially biased inferences from the text, and providing a clear navigation path for complex queries.

**Enhance reasoning capabilities**: The structure of the graph naturally supports multi-hop reasoning. The system can traverse along the path in the graph and discover indirect but key knowledge associations, thereby answering complex questions that require multi-step logical derivation.

**Factuality and Interpretability**: The answer based on knowledge graph retrieval can trace its reasoning path in the graph, which provides a factual basis for the answer, greatly enhances the interpretability and credibility of the system, and effectively suppresses hallucinations.

**Heterogeneous data integration**: Knowledge graphs can seamlessly integrate structured (such as databases) and unstructured (such as text) data from different sources to form a comprehensive and unified knowledge view, which is particularly important in fields such as finance and medical care that require the integration of massive heterogeneous information.

Further:

- **Semantic Schema and Ontology Support**: Clarify entity types, relationship constraints, and attribute domain values ​​through Schema and Ontology, constrain the reasoning space, and reduce ambiguity.
- **Tracing and Confidence Management**: Edges and nodes can carry sources, timestamps and confidences to facilitate source attribution and conflict resolution during generation.
- **Time and Version Modeling**: Supports time-state edges and versioned nodes, allowing querying of "historical status" and "valid interval" (Time-travel Query).

### 1.3 GraphRAG: A paradigm innovation

GraphRAG changes the retrieval target from independent text fragments to finding related entities, relationships, paths or subgraphs in the knowledge graph. This transformation fundamentally enhances the capabilities of the RAG system. Existing research and practice reports show that GraphRAG has significant advantages over traditional RAG in key indicators such as context recall, multi-hop question and answer accuracy, factual consistency and interpretability. This marks the paradigm evolution of RAG technology from "information retrieval" to "knowledge utilization".

## 2. The core architecture and workflow of the GraphRAG framework

Most GraphRAG frameworks follow a common three-stage process: knowledge graph construction, graph retrieval and enhancement generation.

### 2.1 Three stages of general architecture

**Knowledge graph construction** is the foundation of GraphRAG. The goal of this stage is to build a high-quality knowledge graph from raw data:

- Knowledge extraction: Use LLM or IE pipeline to extract entities, relationships and attributes from un/semi-structured text, including reference resolution, alias normalization (Normalization) and terminology standardization.
- Quality control: Confidence assessment, human-machine collaborative sampling and conflict resolution for triples; consistency verification through rules and ontology constraints when necessary.
- Graph fusion: Perform entity alignment and deduplication (Entity Resolution), merge cross-source knowledge and retain traceability information such as source/timestamp.
- Storage and indexing: Implement the graph database (such as Neo4j/NebulaGraph/TigerGraph/Neptune, etc.) and establish necessary attribute and relationship indexes to facilitate subsequent efficient retrieval and traversal.

**Graph retrieval** stage, when the user puts forward a query, the system no longer simply performs a vector similarity search, but performs a more complex graph retrieval operation. Hybrid search strategies are the mainstream trend:

- Entity positioning: First use entity linking or vector retrieval to lock the core entity nodes in the graph.
- Subgraph exploration: Starting from the hit node, use graph query language (such as Cypher) or traversal algorithm to perform neighborhood expansion, path discovery and constraint filtering (such as limiting the relationship type, hop count, time interval and confidence threshold).
- Structured evidence extraction: Serialize path and key node attributes into readable evidence fragments, or generate text summaries of subgraphs.
- Advanced retrieval: For example, GraphRAG uses community detection (such as Leiden) to generate multi-level abstracts to achieve "global-local" joint retrieval.

Example (Cypher): Query the acquisition path and target industry within two hops of company A.

```cypher
MATCH (a:Company {name: "A"})-[:ACQUIRED]->(b:Company)-[:IN_INDUSTRY]->(i:Industry)
RETURN a.name AS acquirer, b.name AS target, i.name AS industry
UNION
MATCH (a:Company {name: "A"})-[:ACQUIRED]->(:Company)-[:ACQUIRED]->(b2:Company)-[:IN_INDUSTRY]->(i2:Industry)
RETURN a.name AS acquirer, b2.name AS target, i2.name AS industry;
```

**Augmented generation** is the final step, in which the retrieved structured knowledge (such as attributes of related entities, relationship paths between entities, text summaries describing subgraphs, etc.) is injected into the LLM prompt (Prompt) together with the original query. Practical points:

- Prompt design: explicitly require the model to cite "graph evidence" and give the source and path in the answer (can be used as an optional citation appendix).
- Evidence fusion: jointly provide "structured triples/paths" and "original text fragments", taking into account both facts and text details.
- Constrained answer: For sensitive fields (finance/medical), templated output and field verification can be used to reduce illusions.

### 2.2 Methodological classification

According to the review research, existing GraphRAG methods can be roughly classified into three categories, reflecting the differences in the depth of the combination of knowledge graphs and RAG.

**Knowledge-driven:** The retrieval process mainly or completely relies on the knowledge graph. This type of method directly performs query and reasoning on the graph (such as text conversion to Cypher/Gremlin), and is suitable for tasks that require strong logical constraints and interpretability.

**Index-driven:** Integrate the structural information of the knowledge graph into the text index. For example, using neighbor entities/relationships as metadata features, or splicing sub-image summaries into text and then vectorizing them to improve recall and rearrangement effects.

**Mixed type:** Use graph retrieval and text retrieval at the same time, and uniformly rearrange and fuse the results:

- Simple fact questions and answers can prioritize image retrieval; narrative/descriptive questions can supplement textual evidence.
- For complex queries, you can first locate the path on the map, and then use the path nodes as "navigation hubs" for targeted document retrieval.

Comparison of pros and cons (overview):

- Knowledge-driven: high accuracy/high interpretability, but coverage and fault tolerance are limited by the completeness of the graph.
- Index-driven: low integration cost, but weak explicit reasoning and average interpretability.
- Hybrid type: The overall performance is more robust, but the system complexity and engineering cost are higher.

## 3. Introduction to the cutting-edge GraphRAG framework (as of 2025)

In recent years, a number of representative GraphRAG frameworks have emerged in academia and industry, each with its own focus on architectural design and application scenarios.

### 3.1 GraphRAG (Microsoft)

GraphRAG is a heavyweight framework whose core idea is to "build global knowledge first and then retrieve it on demand" [^1][^2]. Typical process:

- Text→Triple/Subgraph construction→Graph (attribute graph);
- Use community detection (such as Leiden) to divide the graph and generate global/community/local summaries by level;
- When querying, first match the global/community summary, and then drill down to local subgraphs and original text evidence;
- Return answers from dual perspectives of "overall view" and "evidence path" as needed.

The advantage of this method is that it provides strong global awareness and good interpretability, and is suitable for exploratory analysis, panoramic summary and multi-hop question and answer; however, the cost of early construction and hierarchical summary is high, and batch processing/incremental update and caching strategies need to be designed for scenarios where data continues to change.

### 3.2 LightRAG

In response to the "weight" problem of GraphRAG, LightRAG aims to be lightweight, efficient, and easy to expand[^3][^4]:

- The common core ideas are "double-layer retrieval" and "graph-enhanced indexing", which achieve both global and local retrieval capabilities at a lower construction cost;
- Put more emphasis on embedding structural signals into text indexing and rearrangement, weakening the complex community discovery process;
- Suitable for scenarios with limited resources, rapid iteration and online updates.

### 3.3 FRAG (Flexible RAG)

FRAG emphasizes "flexibility" and "modularity" to adapt to queries of different complexity[^5]:

- Query offloading: Determine requests as simple/complex through classifiers or rules;
- Simple query: direct entity link + attribute search, low-latency return;
- Complex queries: Activate the path retrieval/multi-hop reasoning module, and then integrate text evidence;
- Easy to expand: modular design and reuse of pre-trained LLM capabilities, low migration cost.

### 3.4 GraphIRAG (Iterative Knowledge Retrieval)

GraphIRAG introduces the idea of ​​"iterative retrieval" and believes that a single retrieval is not enough to solve complex problems[^6]:

- The controller triggers multiple rounds of graph query during generation, gradually completing the evidence chain;
- Use "new information gain" as the criterion to decide whether to continue iteration and when to stop;
- More robust to time-sensitive and multi-hop reasoning tasks.

## 4. Performance evaluation and benchmark testing

### 4.1 Core evaluation indicators

GraphRAG's evaluation system is multi-dimensional and mainly covers three aspects.

**Search quality**:

- Traditional: Precision, Recall, F1, Hit Rate@K.
- RAG specific:
- Context Precision = the proportion of "really relevant" in the retrieved context;
- Context Recall = the proportion of all “gold standard evidence” that should support the answer being retrieved;
- Citation/Attribution Accuracy (Citation/Attribution) = Whether the key assertions in the generated answer are correctly supported by the retrieved evidence.

**Build Quality**:

- Q&A: Exact Match (EM), F1;
- Summary: ROUGE;
- Factual consistency/fidelity (Faithfulness): The consistency of assertions and evidence, which can be combined with "sentence-by-sentence marking + attribution checking".

**System Performance**:

Inference latency (Latency) is the total time required from receiving a query to returning the final answer. It is a core indicator of system response speed. Throughput refers to the number of queries (QPS) that the system can process per unit time. Cost includes the number of API calls, computing resources (GPU/CPU), memory consumption, etc.

### 4.2 Commonly used benchmark data sets

The academic community has established a series of benchmark data sets to evaluate GraphRAG's capabilities on different tasks.

**Multi-hop Q&A** The data set requires the model to reason between multiple knowledge sources, which is the touchstone for testing GraphRAG's core capabilities. Representative datasets include HotpotQA, 2WikiMultihopQA, and MuSiQue.

**Complex Question Answering** Datasets contain queries with complex structures. Representative data sets include WebQSP and ComplexWebQuestions (CWQ).

**Knowledge graph-driven QA** is a data set specifically designed for evaluating the question-answering capabilities of knowledge graphs, such as KGQAgen-10k.

However, some studies have pointed out that some existing benchmarks (such as WebQSP) have data quality issues or the evaluation methods are too rigid (such as strict EM), which may lead to an underestimation of model performance. Therefore, building higher-quality and more scientific benchmarks is an important direction in the future.

## 5. Production environment deployment practices and challenges

Moving the GraphRAG framework from the lab to production comes with a unique set of engineering and technical challenges.

- **Construction and dynamic maintenance of knowledge graph**: The construction of high-quality knowledge graph itself is a time-consuming and labor-intensive knowledge project. In a production environment, knowledge needs to be kept up-to-date. How to design an efficient, accurate, and low-cost dynamic update mechanism is a core difficulty.

- **System performance and scalability**: As the size of the knowledge base and user query volume increase, the system's response delay, throughput and resource consumption have become major bottlenecks. Production deployments need to deal with specific issues such as slow vectorization, memory overflows, and API failures.

- **Security and Privacy Protection**: The RAG system introduces external data sources, which brings new security risks, including data privacy leaks, malicious data injected into the model (model poisoning), and attacks against the retrieval module.

- **Cost Control**: Large-scale deployment of the GraphRAG system requires a large amount of computing and storage resources. How to optimize the system architecture to reduce operating costs is an important business consideration.

## References

[^1]: [Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130)

[^2]: [Microsoft GraphRAG Documentation](https://microsoft.github.io/graphrag/)

[^3]: [Guo et al. (2024). *LightRAG: Simple and Fast Retrieval-Augmented Generation*](https://arxiv.org/abs/2410.05779)

[^4]: [HKUDS LightRAG Repository](https://github.com/HKUDS/LightRAG)

[^5]: [Zhang et al. (2025). *FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs*](https://arxiv.org/abs/2501.09957)

[^6]: [Chen et al. (2025). *GraphIRAG: A Knowledge Graph-Based Iterative Retrieval-Augmented Generation Framework for Temporal Reasoning*](https://arxiv.org/abs/2503.14234)
