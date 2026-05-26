# Section 1 Introduction to RAG

## 1. What is RAG?

### 1.1 Core Definition

In essence, RAG (Retrieval-Augmented Generation) is a technical paradigm designed to solve the problem of "knowing what is happening but not why" in large language models (LLM). Its core is to combine the "**parametric knowledge**" learned inside the model (the solidified, fuzzy "memory" in the model weights) with the "**non-parametric knowledge**" (accurate, external data that can be updated at any time) from the external knowledge base. Its operating logic is that before LLM generates text, it first dynamically obtains relevant information from the external knowledge base through a retrieval mechanism, and integrates these "reference materials" into the generation process, thereby improving the accuracy and timeliness of the output [^1] [^2] [^3].

> 💡 **Summary in one sentence**: RAG allows LLM to learn "open book examination". It can not only use the knowledge it has learned, but also check external materials at any time.

### 1.2 Technical principles

So, how does the RAG system achieve the combination of "parametric knowledge" and "non-parametric knowledge"? As shown in Figure 1-1, its architecture mainly completes this process through two stages:

(1) **Retrieval stage: looking for “non-parametric knowledge”**
- **Knowledge vectorization**: **Embedding Model** plays the role of "connector". It encodes the external knowledge base into a vector index (Index) and stores it in the **vector database**.
- **Semantic Recall**: When the user initiates a query, the retrieval module uses the same embedding model to vectorize the question, and through **Similarity Search**, it accurately locates the document fragments most relevant to the question from the massive data.

(2) **Generation stage: merging two kinds of knowledge**
- **Context integration**: **Generation module** receives relevant document fragments sent during the retrieval phase and the user's original questions.
- **Instruction Guided Generation**: This module will follow the preset **Prompt** instructions to effectively integrate context and questions, and guide LLM (such as DeepSeek) to generate controllable and well-founded text.

<div align="center">
<img src="./images/1_1_1.svg" width="60%" alt="RAG two-stage architecture diagram">
<p>Figure 1-1 Schematic diagram of RAG two-stage architecture</p>
</div>

### 1.3 Technology evolution classification

RAG's technical architecture has experienced an evolution from simplicity to complexity, which can be roughly divided into three stages as shown in Figure 1-2 [^4].

<div align="center">
<img src="./images/1_1_2.png" width="80%" alt="RAG Technology Evolution Classification">
<p>Figure 1-2 RAG technology evolution classification</p>
</div>

The specific comparison of these three stages is shown in Table 1-1.

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;"></th>
<th style="text-align: center;">Naive RAG</th>
<th style="text-align: center;">Advanced RAG</th>
<th style="text-align: center;">Modular RAG</th>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Process</strong></td>
<td style="text-align: center;"><strong>Offline:</strong> <code>Index</code><br><strong>Online:</strong> <code>Retrieve → Generate</code></td>
<td style="text-align: center;"><strong>Offline:</strong> <code>Index</code><br><strong>Online:</strong> <code>...→ Before retrieval → ... → After retrieval → ...</code></td>
<td style="text-align: center;">Building block-style orchestration process</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Features</strong></td>
<td style="text-align: center;">Basic linear process</td>
<td style="text-align: center;">Add <strong>optimization steps before and after retrieval</strong></td>
<td style="text-align: center;">Modular, combinable, dynamically adjustable</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Key technologies</strong></td>
<td style="text-align: center;">Basic vector retrieval</td>
<td style="text-align: center;"><strong>Query Rewrite</strong><br><strong>Result Rerank</strong></td>
<td style="text-align: center;"><strong>Dynamic Routing</strong><br><strong>Query Transformation</strong><br><strong>Multiple Fusion (Fusion)</strong></td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Limitations</strong></td>
<td style="text-align: center;">The effect is unstable and difficult to optimize</td>
<td style="text-align: center;">The process is relatively fixed and the optimization points are limited</td>
<td style="text-align: center;">High system complexity</td>
  </tr>
</table>
<p><em>Table 1-1 RAG technology evolution classification comparison</em></p>
</div>

> "Offline" refers to data preprocessing work (such as index construction) completed in advance; "online" refers to the real-time processing process after the user initiates a request.

## 2. Why use RAG?

### 2.1 Technology selection: RAG vs. fine-tuning

When choosing a specific technology path, an important consideration is the balance between costs and benefits. Usually, we should give priority to the solution with the smallest changes to the model and the lowest cost, so the technology selection path often follows the order of **Prompt Engineering (Prompt Engineering) -> Retrieval Enhancement Generation -> Fine-tuning**.

We can understand the difference between these technologies from two dimensions. As shown in Figure 1-3, the horizontal axis represents "LLM optimization", that is, the extent to which the model itself is modified. From left to right, the degree of optimization increases, with hint engineering and RAG not changing the model weights at all, and fine-tuning directly modifying the model parameters. **The vertical axis represents "contextual optimization"**, which is the extent to which the information input to the model is enhanced. From bottom to top, the degree of enhancement is getting higher and higher, where prompt engineering only optimizes the way of asking questions, while RAG greatly enriches contextual information by introducing external knowledge bases.

<div align="center">
<img src="./images/1_1_3.svg" width="60%" alt="Technical selection path" />
<p>Figure 1-3 Selection path chart</p>
</div>

Based on this, our choice path is clear:
- **Try the prompt project first**: Guide the model through carefully designed prompt words, which is suitable for scenarios where the task is simple and the model already has relevant knowledge.
- **RAG then**: If the model lacks specific or real-time knowledge to answer, use RAG to provide it with contextual information through a plug-in knowledge base.
- **Final consideration for fine-tuning**: Fine-tuning is the final and most appropriate option when the goal is to change "how" the model does (behavior/style/format) rather than "what" it knows (knowledge). For example, let the model learn to strictly follow a unique output format, imitate the conversational style of a specific person, or "distill" extremely complex instructions into the model weights.

The emergence of RAG bridges the gap between general models and specialized fields. It is especially effective in solving the limitations of LLM as shown in Table 1-2:

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
<th style="text-align: center;">Question</th>
<th style="text-align: center;">RAG’s solution</th>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Static knowledge limitations</strong></td>
<td style="text-align: center;">Retrieve external knowledge base in real time and support dynamic updates</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Hallucination</strong></td>
<td style="text-align: center;">Generated based on retrieved content, reducing error rate</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Insufficient expertise in the field</strong></td>
<td style="text-align: center;">Introducing domain-specific knowledge bases (such as medical/legal)</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Data Privacy Risk</strong></td>
<td style="text-align: center;">Localized deployment of knowledge base to avoid leakage of sensitive data</td>
  </tr>
</table>
<p><em>Table 1-2 RAG’s solutions to LLM limitations</em></p>
</div>

### 2.2 Key advantages

(1) **Double improvement of accuracy and credibility**

The core value of RAG is to break through the limitations of model pre-training knowledge. It can not only supplement knowledge blind spots in professional fields, but also effectively suppress the illusion of "serious nonsense" by providing specific reference materials. Thesis research also shows that the content generated by RAG is also significantly better than pure LLM in terms of **specificity** and **diversity**. More importantly, RAG has **traceability** - each answer can be found in the corresponding original document source. This "well-documented" feature greatly improves the credibility of the content in serious scenarios such as legal and medical.

(2) **Timeliness Guarantee**

In terms of knowledge updating, RAG solves the **knowledge lag problem** inherent in LLM (that is, the model does not know what happened after the training deadline). RAG allows the knowledge base to be dynamically updated independently of the model - new policies or new data can be retrieved as soon as they are entered into the database. This ability is called "Index Hot-swapping" in the paper - just like changing a memory card to a robot, it can instantly switch its world knowledge base without retraining the model, realizing real-time online knowledge.

(3) **Significant comprehensive cost benefit**

From an economic perspective, RAG is a cost-effective solution. First of all, it avoids the huge computing power cost caused by high-frequency fine-tuning; secondly, due to the powerful assistance of external knowledge, when we deal with problems in specific fields, we can often use basic models with smaller parameters to achieve similar effects, thus directly reducing the cost of reasoning. This architecture also reduces the consumption of computational resources required to try to "cram" massive amounts of knowledge into model weights.

(4) **Flexible modular scalability**

RAG's architecture is extremely inclusive and supports **multi-source integration**. Whether it is PDF, Word or web page data, it can be unified into the knowledge base. At the same time, its **modular design** achieves the decoupling of retrieval and generation, which means that we can independently optimize the retrieval component (such as replacing a better Embedding model) without affecting the stability of the generation component, which facilitates long-term iteration of the system.

### 2.3 Applicable scenario risk classification

Table 1-3 shows the applicability of RAG technology in scenarios with different risk levels.

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
<th style="text-align: center;">Risk level</th>
<th style="text-align: center;">Case</th>
<th style="text-align: center;">RAG applicability</th>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Low risk</strong></td>
<td style="text-align: center;">Translation/grammar check</td>
<td style="text-align: center;">High reliability</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>Medium risk</strong></td>
<td style="text-align: center;">Contract drafting/legal consulting</td>
<td style="text-align: center;">Requires manual review</td>
  </tr>
  <tr>
<td style="text-align: center;"><strong>High risk</strong></td>
<td style="text-align: center;">Evidence analysis/visa decision-making</td>
<td style="text-align: center;">Strict quality control mechanism is required</td>
  </tr>
</table>
<p><em>Table 1-3 RAG applicable scenario risk classification</em></p>
</div>

## 3. How to get started with RAG?

### 3.1 Basic tool chain selection

Building a RAG system usually involves the selection of several key aspects. In **development mode**, we can use mature frameworks such as **LangChain** or **LlamaIndex** for rapid integration, **or you can choose native development** that does not rely on frameworks to gain more granular control over the system process (this is not difficult with the assistance of AI programming). In terms of **memory carrier** (vector database), there are solutions suitable for large-scale data such as **Milvus** and **Pinecone**, as well as lightweight or localized options such as **FAISS** and **Chroma**, which need to be flexibly decided according to the specific business scale. In order to quantify the effect later, automated **assessment tools** such as **RAGAS** or **TruLens** can also be introduced.

### 3.2 Four steps to build a minimum viable system (MVP)

(1) **Data preparation and cleaning**: This is the foundation of the system. We need to standardize multi-source heterogeneous data such as PDF and Word, and adopt reasonable **chunking strategies** (such as splitting by semantic paragraphs instead of fixed number of characters) to avoid information being fragmented during cutting.

(2) **Index construction**: Convert the segmented text into vectors through **embedding model** and store them in the database. **Metadata** (such as source, page numbers) can be associated at this stage, which is helpful for precise citations later.

(3) **Search strategy optimization**: Don’t rely on a single vector search. **Hybrid retrieval** (vector + keyword) and other methods can be used to improve the recall rate, and the **reordering** model can be introduced to re-select the retrieval results to ensure that LLM sees only the best.

(4) **Generation and Prompt Engineering**: Finally, design a set of clear **Prompt templates** to guide LLM to answer user questions based on the retrieved context, and clearly require the model to "say you don't know if you don't know" to prevent hallucinations.

### 3.3 Novice friendly solution

If you want to quickly verify your ideas instead of digging into the code, you can try visual knowledge base platforms like **FastGPT** or **Dify**. They encapsulate the complex RAG process and only need to upload documents to use. For developers, using open source templates such as **LangChain4j Easy RAG** ​​or **TinyRAG** ​​[^6] on GitHub is also an efficient way to get started.

### 3.4 Advancement and Challenges

After the basic RAG system is built, the next step towards advancement will focus on how to evaluate, diagnose and break through its inherent bottlenecks.

(1) **Assessment Dimensions and Challenges**

The quality of a RAG system cannot be determined by feeling alone. The industry usually conducts quantitative evaluations from several dimensions, firstly **retrieval relevance** (whether the found content contains answers), and secondly **generation quality**, which can be subdivided into **semantic accuracy** (whether the meaning of the answer is correct) and **lexical matching** (whether professional terms are used appropriately).

These evaluation dimensions also directly correspond to the main challenges currently facing RAG. For example, the problem of **retrieval dependence** - if the retrieval system recalls wrong information, no matter how strong the LLM is, it will "seriously talk nonsense". In addition, common RAG architectures also generally struggle with multi-hop reasoning problems that require comprehensive analysis across multiple documents.

(2) **Optimization direction and architecture evolution**

In response to the above challenges, the community has explored a variety of optimization paths. At the **performance level**, efficiency and capability boundaries can be improved through **index tiering** (enabling caching for high-frequency data) and **multimodal extension** (supporting image/table retrieval). At the **architectural level**, simple linear processes are being replaced by more complex **design patterns**. For example, the system can process multiple retrievals in parallel through **branch mode**, or self-correct through **loop mode**. These flexible architectures are the only way to a smarter RAG.

## 4. RAG is dead?

As the ability to use long context windows for large models improves, “RAG is dead” voices are beginning to appear in the community. This argument mainly comes from two aspects. First, it is believed that long context has been able to violently "digest" massive texts, and complex retrieval systems are no longer needed; second, it is criticized that the term RAG itself is too broad, blurring too many technical details, but hindering understanding and optimization.

These views ignore the universal laws in the evolution of a technical concept. Just as we can easily give modern complex RAG systems a more precise and more deceptive name, such as "Large Language Model Knowledge Management Expert System (LKE)". Because it has already gone beyond the simple scope of the original "retrieval-enhancement-generation". But this "name-changing game" just illustrates the superficiality of the "RAG is dead" theory - it is tantamount to using a new bottle to hold the aging wine of RAG.

> The author is not trying to create a new word here, but why the name LKE? It represents three core elements:
> - **L (Large Language Model)**: Emphasizes that the driving force of the system is the large language model.
> - **K (Knowledge Management)**: It means that the system is like a knowledge manager, accurately finding the knowledge we need (**retrieval**), and assisting us in subsequent use of large models for higher-order applications.
> - **E (Expert)**: It means that the system can, like an expert, go through a series of steps such as routing, analysis, fusion, correction, etc., and finally give answers (**generate**) and solve problems.

It can be compared to **Transformer**. Today, whether it is Decoder-only represented by GPT or Encoder-only represented by BERT, we are accustomed to calling it "Transformer-based architecture", although they are very different from the complete form in the original paper. But the label Transformer captured a core leap in a technological paradigm and became a symbol of a technological era. In the same way, the core of **RAG is to "combine the intrinsic parametric knowledge of LLM with the external non-parametric knowledge"**. As long as this idea or demand remains unchanged, no matter how many modules we add to it - query transformation, multi-way recall or self-correction, etc., it is still essentially an evolution under this framework.

Therefore, "RAG is dead" is a false proposition. On the contrary, **RAG is alive and well as a concept**, and it is becoming, like Transformer, an infrastructure paradigm that continues to absorb new technologies and evolve. Its vitality lies in its "beyond all recognition" and "all-encompassing". The goal of this tutorial is to draw a clear map that depicts the whole picture of RAG. When we can deconstruct each of its modules and understand every possibility of it, whether it is RAG or LKE, it does not matter**. What we have to do is to learn and expand (combining LLM's intrinsic parametric knowledge with external non-parametric knowledge) through the classic example of RAG to solve problems of this type.

> RAG technology is still developing rapidly, and you can continue to pay attention to the latest developments in academia and industry!

## References

[^1]: [Genesis, J. (2025). *Retrieval-Augmented Text Generation: Methods, Challenges, and Applications*](https://www.researchgate.net/publication/391141346_Retrieval-Augmented_Generation_Methods_Applications_and_Challenges).

[^2]: [Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*](https://arxiv.org/abs/2312.10997).

[^3]: [Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401). 

[^4]: [Gao et al. (2024). *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*](https://arxiv.org/abs/2407.21059).

[^6]: [*TinyRAG: GitHub project*](https://github.com/KMnO4-zx/TinyRAG).