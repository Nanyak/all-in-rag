# Section 1 Data Loading

Although the contents of this section are very important in practical applications, due to the iterative updates of various document loaders and the different needs of various AI applications, the specific selection needs to be based on the actual situation. This section is only a brief introduction, but please be sure to **pay attention to the data loading** link, **"Garbage In, Garbage Out"** - high-quality input is the prerequisite for high-quality output.

## 1. Document loader

### 1.1 Main functions

In the RAG system, **data loading** is the first and indispensable step in the entire pipeline. The document loader is responsible for converting unstructured documents in various formats (such as PDF, Word, Markdown, HTML, etc.) into structured data that can be processed by the program. The quality of data loading will directly affect subsequent index construction, retrieval results and final generation quality.

The document loader generally needs to complete three core tasks in RAG's data pipeline. One is to parse original documents in different formats and convert PDF, Word, Markdown The second is to extract key information such as document source, page number, author, etc. as metadata during the parsing process. The third is to organize the text and metadata into a unified data structure to facilitate subsequent segmentation, vectorization, and storage. The overall process is similar to the extraction, conversion, and loading in traditional data engineering. The goal is to clean and align messy original documents into standardized corpus suitable for retrieval and modeling.

### 1.2 Current mainstream RAG document loader

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
<th style="text-align: center;">Tool name</th>
<th style="text-align: center;">Features</th>
<th style="text-align: center;">Applicable scenarios</th>
<th style="text-align: center;">Performance</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>PyMuPDF4LLM</strong></td>
<td style="text-align: center;">PDF→Markdown conversion, OCR+table recognition</td>
<td style="text-align: center;">Scientific research documents, technical manuals</td>
<td style="text-align: center;">Open source and free, GPU accelerated</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>TextLoader</strong></td>
<td style="text-align: center;">Basic text file loading</td>
<td style="text-align: center;">Plain text processing</td>
<td style="text-align: center;">Lightweight and efficient</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>DirectoryLoader</strong></td>
<td style="text-align: center;">Batch directory file processing</td>
<td style="text-align: center;">Mixed format document library</td>
<td style="text-align: center;">Supports multi-format expansion</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Unstructured</strong></td>
<td style="text-align: center;">Multiple format document parsing</td>
<td style="text-align: center;">PDF, Word, HTML, etc.</td>
<td style="text-align: center;">Unified interface, intelligent analysis</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>FireCrawlLoader</strong></td>
<td style="text-align: center;">Web content capture</td>
<td style="text-align: center;">Online documents, news</td>
<td style="text-align: center;">Real-time content acquisition</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>LlamaParse</strong></td>
<td style="text-align: center;">Deep PDF structure analysis</td>
<td style="text-align: center;">Legal contracts, academic papers</td>
<td style="text-align: center;">High parsing accuracy, commercial API</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Docling</strong></td>
<td style="text-align: center;">Modular enterprise-level parsing</td>
<td style="text-align: center;">Corporate contracts and reports</td>
<td style="text-align: center;">IBM ecological compatibility</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Marker</strong></td>
<td style="text-align: center;">PDF→Markdown, GPU accelerated</td>
<td style="text-align: center;">Scientific research documents and books</td>
<td style="text-align: center;">Focus on PDF conversion</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>MinerU</strong></td>
<td style="text-align: center;">Multimodal integrated analysis</td>
<td style="text-align: center;">Academic literature, financial statements</td>
<td style="text-align: center;">Integrated LayoutLMv3+YOLOv8</td>
  </tr>
</table>
<p><em>Table 2-1 Current mainstream RAG document loaders</em></p>
</div>

## 2. Unstructured document processing library

### 2.1 Core advantages of Unstructured

**Unstructured** [^1] is a professional document processing library specifically designed for unstructured data preprocessing in RAG and AI fine-tuning scenarios. It provides a unified interface to handle multiple document formats and is one of the most widely used document loading solutions. Unstructured has obvious advantages in format support and content parsing. On the one hand, it supports multiple document formats such as PDF, Word, Excel, HTML, Markdown, etc., and avoids writing separate codes for different formats through a unified API interface. On the other hand, it can automatically identify document structures such as titles, paragraphs, tables, and lists, while retaining corresponding metadata information.

<div align="center">
<img src="./images/2_1_1.png" width="80%" alt="Unstructured official website interface">
<p>Figure 2-1 Unstructured official website interface</p>
</div>

### 2.2 Supported document element types

Unstructured can identify and classify the following document elements [^2]:

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
<th style="text-align: center;">Element type</th>
<th style="text-align: center;">Description</th>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Title</code></td>
<td style="text-align: center;">Document title</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>NarrativeText</code></td>
<td style="text-align: center;">Body text consisting of multiple complete sentences, excluding titles, headers, footers, and captions</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>ListItem</code></td>
<td style="text-align: center;">List items, body text elements belonging to the list</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Table</code></td>
<td style="text-align: center;">Table</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Image</code></td>
<td style="text-align: center;">Image metadata</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Formula</code></td>
<td style="text-align: center;">Formula</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Address</code></td>
<td style="text-align: center;">Physical address</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>EmailAddress</code></td>
<td style="text-align: center;">Email address</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>FigureCaption</code></td>
<td style="text-align: center;">Picture title/caption</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Header</code></td>
<td style="text-align: center;">Document header</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>Footer</code></td>
<td style="text-align: center;">Document footer</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>CodeSnippet</code></td>
<td style="text-align: center;">Code snippet</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>PageBreak</code></td>
<td style="text-align: center;">Page separator</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>PageNumber</code></td>
<td style="text-align: center;">Page number</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>UncategorizedText</code></td>
<td style="text-align: center;">Uncategorized free text</td>
  </tr>
  <tr>
    <td style="text-align: center;"><code>CompositeElement</code></td>
<td style="text-align: center;">Composite elements generated during block processing*</td>
  </tr>
</table>
<p><em>Table 2-2 Document element types supported by Unstructured</em></p>
</div>

>`CompositeElement`is a special element type generated through chunking processing, which is composed of one or more consecutive text elements. For example, multiple list items might be combined into a single block.

## 3. From LangChain encapsulation to original Unstructured

In the example in Chapter 1, we used LangChain’s`UnstructuredMarkdownLoader`, which is LangChain’s encapsulation of the Unstructured library. Next I show you how to use the Unstructured library directly, which gives you greater flexibility and control.

> [Complete code of this section](https://github.com/datawhalechina/all-in-rag/blob/main/code/C2/01_unstructured_example.py)

### 3.1 Code Example

Create a simple example that attempts to load and parse a PDF file using the Unstructured library.

```python
from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

> If an error`ImportError: libgl.so.1 cannot open shared object file no such file or directory`occurs when running the code, execute`sudo apt-get install python3-opencv`to install the dependent libraries.

**partition function parameter analysis:**

-`filename`: Document file path, supports local file path;
-`content_type`: Optional parameter, specify the MIME type (such as "application/pdf"), which can bypass automatic file type detection;
-`file`: Optional parameter, file object, choose one from filename;
-`url`: Optional parameter, remote document URL, supports direct processing of network documents;
-`include_page_breaks`: Boolean value, whether to include page separators in the output;
-`strategy`: processing strategy, optional "auto", "fast", "hi_res", etc.;
-`encoding`: Text encoding format, automatically detected by default.

The`partition`function uses automatic file type detection and will internally route to the corresponding dedicated function based on the file type (for example, PDF files will call`partition_pdf`). If you need more professional PDF processing, you can directly use`from unstructured.partition.pdf import partition_pdf`, which provides more PDF-specific parameter options, such as OCR language settings, image extraction, table structure reasoning and other advanced functions, and has better performance.

> In practical applications, for pdf processing, models or tools such as PaddleOCR and MinerU are currently more used.

## practise

- Use`partition_pdf`to replace the current`partition`function and try to use`hi_res`and`ocr_only`respectively to analyze and observe the changes in the output results.

## References

[^1]: [*Unstructured Open-Source Documentation*](https://docs.unstructured.io/open-source/)

[^2]: [*Unstructured Open-Source: Document Elements*](https://docs.unstructured.io/open-source/concepts/document-elements)
