# LangChain Retriever 가이드

Retriever는 문자열 질의를 받아 관련 `Document` 목록을 반환하는 컴포넌트다. Vector Store가 문서와 벡터를 저장·검색한다면, Retriever는 검색 방식과 후처리 전략을 표준 실행 인터페이스로 감싼다.

이 문서는 현재 저장소의 LangChain 1.x(`langchain 1.4.2`, `langchain-classic 1.0.8`)와 `retriever.invoke("질문")` 호출 방식을 기준으로 한다.

## 공통 준비

<details>
<summary>공통 준비 코드 보기</summary>

```python
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

docs = [
    Document(page_content="RAG는 검색 결과를 근거로 답변을 생성한다.",
             metadata={"category": "rag", "year": 2025, "doc_id": "rag-1"}),
    Document(page_content="벡터 검색은 의미적으로 유사한 문서를 찾는다.",
             metadata={"category": "search", "year": 2024, "doc_id": "search-1"}),
    Document(page_content="BM25는 검색어가 정확히 일치하는 문서에 강하다.",
             metadata={"category": "search", "year": 2023, "doc_id": "search-2"}),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="retriever-guide"
)
```

</details>

> 실행에는 `OPENAI_API_KEY`가 필요하다. 운영 환경에서는 인덱싱과 질의 단계를 분리해 문서를 중복 적재하지 않는다.

---

## 1. VectorStoreRetriever

### 개념

Vector Store의 검색 기능을 가장 단순한 Retriever로 변환한다. `similarity`는 유사도, `mmr`은 관련성과 결과 다양성, `similarity_score_threshold`는 점수 임계값을 사용한다. 지원 방식과 점수 의미는 Vector Store마다 다를 수 있다.

<details>
<summary>간단 예제 보기</summary>

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)
result = retriever.invoke("의미 기반 검색은 어떻게 하나요?")

# 비슷한 결과가 반복되면 MMR로 다양성을 확보한다.
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5},
)
```

</details>

### 사용성

- **장점:** 설정이 단순하고 빠르며 대부분의 Vector Store에서 쓸 수 있다.
- **제약:** 정확한 키워드, 표현 차이, 복잡한 metadata 조건을 단독으로 모두 처리하기 어렵다.
- **적합:** RAG의 첫 기준선, 의미 검색 중심 FAQ·문서 검색.

### 선택 시 고려사항

- `k`를 작게 시작해 Recall@k와 답변 품질을 보며 늘린다.
- 중복이 많으면 `mmr`, 무관한 결과가 많으면 임계값이나 reranker를 검토한다.
- 임베딩 모델을 바꾸면 기존 문서도 같은 모델로 다시 임베딩한다.

---

## 2. ContextualCompressionRetriever

### 개념

기본 리트리버가 가져온 문서를 질의에 맞게 **필터링, 재정렬하거나 필요한 구절만 추출**하는 래퍼다. 여기서 compression은 파일 압축이 아니라 LLM에 전달할 문맥을 줄인다는 뜻이다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)
result = compression_retriever.invoke("벡터 검색의 장점은 무엇인가요?")
```

</details>

### 사용성

- **장점:** 불필요한 토큰과 잡음을 줄여 근거 집중도를 높인다.
- **제약:** LLM extractor는 비용과 지연이 크고 중요한 원문 문맥을 자를 수 있다.
- **적합:** 검색 문서는 대체로 맞지만 너무 길거나 무관한 부분이 많을 때.

### 선택 시 고려사항

- 목적에 따라 extractor, relevance filter, embedding filter, cross-encoder reranker를 고른다.
- 숫자·표·법률 문구처럼 원문 보존이 중요하면 추출보다 필터링/reranking이 안전하다.
- 후보 수, 모델 호출 수, 품질과 p95 지연 시간을 함께 측정한다.

---

## 3. EnsembleRetriever

### 개념

여러 Retriever의 순위를 가중 Reciprocal Rank Fusion(RRF)으로 합친다. 흔히 정확한 단어 일치에 강한 BM25와 표현이 달라도 의미가 비슷한 문서를 찾는 벡터 검색을 결합한다.

<details>
<summary>간단 예제 보기</summary>

`BM25Retriever`에는 `rank_bm25`가 추가로 필요하다: `pip install rank_bm25`

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3
dense = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.4, 0.6],
    id_key="doc_id",
)
result = ensemble.invoke("BM25와 의미 검색의 차이")
```

</details>

### 사용성

- **장점:** 키워드 검색과 의미 검색의 약점을 상호 보완한다.
- **제약:** 여러 검색기를 운영하고 중복 기준과 가중치를 조정해야 한다.
- **적합:** 고유명사·상품 코드·오류 코드와 자연어 질문이 함께 들어오는 검색.

### 선택 시 고려사항

- `weights`는 대표 질의의 Recall@k, MRR, nDCG로 조정한다.
- 같은 원문의 chunk를 합치려면 안정적인 문서 ID를 `id_key`로 사용한다.
- 모든 검색기가 동일한 권한 및 metadata 필터를 적용해야 한다.

---

## 4. LongContextReorder

### 개념

Retriever가 아니라 **검색 결과 순서를 바꾸는 후처리기**다. 긴 프롬프트 가운데의 정보가 덜 활용되는 lost-in-the-middle 현상을 완화하도록 중요한 문서를 문맥의 앞과 뒤에 배치한다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_community.document_transformers import LongContextReorder

retrieved = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(
    "RAG 검색 품질 개선 방법"
)
reordered = LongContextReorder().transform_documents(retrieved)
context = "\n\n".join(doc.page_content for doc in reordered)
response = llm.invoke(
    f"다음 문맥만 사용해 답하세요.\n\n{context}\n\n질문: RAG 검색 품질 개선 방법은?"
)
```

</details>

### 사용성

- **장점:** 별도 인덱스나 모델 호출이 없어 비용이 거의 없다.
- **제약:** 검색 정확도를 높이거나 토큰을 줄이지는 않는다.
- **적합:** 많은 검색 결과를 긴 컨텍스트 창에 그대로 넣을 때.

### 선택 시 고려사항

- 입력 문서가 관련도순이어야 재배치가 의미 있다.
- 토큰 한도 초과 문제라면 reorder보다 압축, reranking, `k` 조정이 먼저다.
- 실제 긴 문맥 질의로 적용 전후 답변 정확도를 비교한다.

---

## 5. ParentDocumentRetriever

### 개념

작은 child chunk로 검색하되 그 chunk가 속한 더 큰 parent 문서를 반환한다. 작은 chunk의 정밀한 임베딩과 큰 문서의 충분한 문맥을 함께 얻는다. `MultiVectorRetriever`를 상속한 특화 구현으로 parent/child 분할과 ID 연결을 자동화한다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
child_vectorstore = Chroma(
    collection_name="parent-children", embedding_function=embeddings
)

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=InMemoryStore(),
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)
parent_retriever.add_documents(docs)
result = parent_retriever.invoke("의미 검색의 특징")
```

</details>

### 사용성

- **장점:** 검색 정밀도와 주변 문맥 사이의 균형이 좋다.
- **제약:** Vector Store와 parent docstore를 함께 관리해야 하고 parent가 너무 크면 토큰이 낭비된다.
- **적합:** 매뉴얼, 보고서, 논문처럼 찾은 단락의 앞뒤 설명도 필요한 문서.

### 선택 시 고려사항

- child 크기는 검색 단위, parent 크기는 LLM에 전달할 문맥 단위로 따로 평가한다.
- `InMemoryStore`는 프로세스 종료 시 사라지므로 운영에서는 영속 저장소를 쓴다.
- 갱신·삭제 시 child 벡터와 parent 문서를 함께 동기화한다.

---

## 6. MultiQueryRetriever

### 개념

LLM이 질문을 여러 관점의 검색 질의로 바꾸고, 각각 검색한 결과의 중복을 제거해 합친다. 한 번의 벡터 검색이 놓치는 표현 차이를 보완해 재현율을 높인다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
multi_query = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,
)
result = multi_query.invoke("벡터 검색이 키워드 검색보다 좋은가요?")
```

</details>

### 사용성

- **장점:** 짧고 모호한 질문이나 동의어가 많은 도메인의 검색 누락을 줄인다.
- **제약:** 질의 생성과 반복 검색으로 비용·지연·잡음이 늘 수 있다.
- **적합:** 정답 문서는 있지만 사용자 표현과 문서 표현이 달라 자주 검색되지 않을 때.

### 선택 시 고려사항

- 생성 질의 수, 원본 질의 포함 여부, 각 검색의 `k`를 함께 조절한다.
- 전문 도메인은 약어와 의도를 보존하도록 커스텀 프롬프트를 사용한다.
- Recall이 이미 충분하면 compression/reranking이 더 적합하다.

---

## 7. MultiVectorRetriever

### 개념

하나의 원본 문서에 작은 chunk, 요약, 제목, 가상 질문 등 여러 검색용 표현 벡터를 연결하고 결과로는 원본 문서를 반환한다. ParentDocumentRetriever보다 표현을 직접 설계할 수 있는 일반형 도구다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

parents = [
    Document(page_content="RAG 원문 전체: 검색과 생성 단계에 대한 긴 설명..."),
    Document(page_content="검색 원문 전체: BM25와 벡터 검색에 대한 긴 설명..."),
]
parent_ids = ["parent-1", "parent-2"]
representations = [
    Document(page_content="RAG의 검색과 생성 과정", metadata={"doc_id": "parent-1"}),
    Document(page_content="검색 증강 생성이란?", metadata={"doc_id": "parent-1"}),
    Document(page_content="BM25와 dense retrieval", metadata={"doc_id": "parent-2"}),
    Document(page_content="키워드와 의미 검색 비교", metadata={"doc_id": "parent-2"}),
]

representation_store = Chroma(
    collection_name="multi-vector", embedding_function=embeddings
)
parent_store = InMemoryStore()
multi_vector = MultiVectorRetriever(
    vectorstore=representation_store,
    docstore=parent_store,
    id_key="doc_id",
    search_kwargs={"k": 4},
)
representation_store.add_documents(representations)
parent_store.mset(list(zip(parent_ids, parents)))
result = multi_vector.invoke("의미 검색과 단어 검색 비교")
```

</details>

### 사용성

- **장점:** 여러 관점을 인덱싱해 재현율을 높이면서 원본 문맥을 반환한다.
- **제약:** 표현 생성 비용, 벡터 수, 저장 공간과 파이프라인 복잡도가 증가한다.
- **적합:** 문서 하나가 여러 주제를 포함하거나 예상 질문·요약을 미리 만들 수 있을 때.

### 선택 시 고려사항

- 표현 metadata의 `id_key`와 docstore 키가 정확히 일치해야 한다.
- 표현 수보다 서로 다른 검색 의도를 잘 나타내는지가 중요하다.
- 원문 변경 시 연결된 모든 벡터를 갱신·삭제할 ID 정책을 먼저 만든다.

---

## 8. SelfQueryRetriever

### 개념

LLM이 자연어 질문을 **의미 검색 질의**와 **metadata 구조화 필터**로 분리한다. 예를 들어 “2024년 이후 RAG 문서”를 질의 `RAG`와 조건 `year >= 2024`로 바꾼다. 비교 연산과 필터 지원 범위는 Vector Store마다 다르다.

<details>
<summary>간단 예제 보기</summary>

```python
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

fields = [
    AttributeInfo(name="category", description="rag 또는 search", type="string"),
    AttributeInfo(name="year", description="문서 작성 연도", type="integer"),
]
self_query = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="RAG 및 검색 기술 문서",
    metadata_field_info=fields,
    enable_limit=True,
)
result = self_query.invoke("2024년 이후 search 분류 문서 2개를 찾아줘")
```

</details>

### 사용성

- **장점:** 사용자가 필터 문법을 몰라도 자연어로 복합 조건 검색을 한다.
- **제약:** LLM 호출이 필요하고, 스키마 설명이나 Vector Store 지원이 부정확하면 필터도 틀린다.
- **적합:** 날짜, 가격, 지역, 범주 등 metadata 조건이 중요한 검색.

### 선택 시 고려사항

- 필드 이름·타입·설명·허용 값을 실제 스키마와 일치시킨다.
- 날짜 저장 형식을 통일하고 생성된 구조화 질의를 관찰 가능하게 만든다.
- 사용자 권한 필터는 LLM에 맡기지 말고 애플리케이션이 강제로 결합한다.

---

## 9. TimeWeightedVectorStoreRetriever

### 개념

벡터 유사도에 최근성과 선택적인 중요도 점수를 더한다. 최근성은 `(1 - decay_rate) ** 경과 시간(시간)`으로 감소한다. 문서가 검색되면 `last_accessed_at`이 갱신되어 최근에 사용한 기억이 다시 선택될 가능성이 커진다.

<details>
<summary>간단 예제 보기</summary>

```python
from datetime import datetime, timedelta
from langchain_chroma import Chroma
from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document

memory_store = Chroma(
    collection_name="time-memory", embedding_function=embeddings
)
time_retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=memory_store,
    decay_rate=0.01,
    k=2,
    other_score_keys=["importance"],
)

now = datetime.now()
time_retriever.add_documents([
    Document(page_content="사용자는 벡터 검색을 선호한다.",
             metadata={"importance": 0.8})
], current_time=now - timedelta(hours=24))
time_retriever.add_documents([
    Document(page_content="오늘 BM25도 시험하고 싶다고 말했다.",
             metadata={"importance": 0.3})
], current_time=now)

result = time_retriever.invoke("사용자가 원하는 검색 방식")
```

</details>

### 사용성

- **장점:** 의미적 관련성과 최신성을 함께 반영한다.
- **제약:** 일반 지식은 오래됐다는 이유로 중요 문서가 밀릴 수 있고 memory stream도 영속화해야 한다.
- **적합:** 사용자 기억, 최근 이벤트, 에이전트 관찰 기록.

### 선택 시 고려사항

- `decay_rate`가 클수록 오래된 문서가 빠르게 밀린다.
- 추가 중요도 점수의 스케일이 유사도와 최근성을 압도하지 않게 한다.
- 시간대와 datetime 형식을 통일한다. 권위가 최신성보다 중요한 규정·계약 문서에는 신중히 쓴다.

---

## 한눈에 비교

| 도구 | 핵심 문제 | 추가 LLM 호출 | 상대 지연 | 대표 사용처 |
|---|---|---:|---:|---|
| VectorStoreRetriever | 기본 의미 검색 | 없음 | 낮음 | RAG 기준선 |
| ContextualCompressionRetriever | 긴 결과와 잡음 | 방식에 따라 다름 | 중간~높음 | 관련 구절 추출·필터·재정렬 |
| EnsembleRetriever | 한 검색 방식의 편향 | 구성에 따라 다름 | 중간 | 키워드+의미 검색 |
| LongContextReorder | 긴 문맥 중간의 정보 손실 | 없음 | 매우 낮음 | 검색 결과 후처리 |
| ParentDocumentRetriever | 정밀 검색과 주변 문맥의 충돌 | 없음 | 낮음~중간 | 긴 문서의 단락 검색 |
| MultiQueryRetriever | 표현 차이로 인한 누락 | 있음 | 높음 | 모호한 질문·동의어 |
| MultiVectorRetriever | 한 문서의 다양한 관점 | 인덱싱 방식에 따라 다름 | 낮음~중간 | 요약·질문·chunk 인덱싱 |
| SelfQueryRetriever | 자연어 metadata 조건 | 있음 | 중간 | 날짜·범주·가격 필터 |
| TimeWeightedVectorStoreRetriever | 관련성+최신성 | 없음 | 낮음~중간 | 대화 기억·최근 이벤트 |

## 리트리버 선택 시 고려사항

### 문제 증상부터 구분하기

- **정답 문서가 검색되지 않는다:** MultiQuery 또는 Ensemble
- **정답 문서의 순위가 낮다:** Ensemble, reranker, 검색 파라미터 조정
- **결과가 너무 길고 잡음이 많다:** ContextualCompression
- **작은 chunk는 찾지만 문맥이 부족하다:** ParentDocument
- **한 문서를 여러 관점으로 찾아야 한다:** MultiVector
- **질문에 날짜·범주 조건이 있다:** SelfQuery
- **최근 정보가 더 중요하다:** TimeWeighted
- **긴 컨텍스트의 중간 정보가 무시된다:** 마지막 단계에 LongContextReorder

### 품질과 운영 비용 함께 평가하기

- **검색:** Recall@k, Precision@k, MRR, nDCG
- **답변:** 정답성, 근거 충실성, 인용 정확성
- **운영:** p50/p95 지연, 질의당 비용, 인덱스 크기, 장애 지점
- **보안:** 사용자별 권한 필터, 테넌트 격리, 민감 metadata 노출
- **유지보수:** 수정·삭제 시 Vector Store와 docstore 동기화

### 단순한 기준선부터 추가하기

```text
VectorStoreRetriever
        ↓ 재현율이 부족하면
MultiQuery 또는 Ensemble
        ↓ 결과가 많거나 잡음이 크면
Compression / Reranking
        ↓ 긴 문맥을 그대로 사용하면
LongContextReorder
```

모든 기능을 한꺼번에 조합하면 어느 단계가 품질을 바꿨는지 알기 어렵다. 고정된 평가 질의와 정답 문서 집합을 만든 뒤 한 요소씩 추가한다.

## 자주 쓰는 조합

### 균형 잡힌 하이브리드 RAG

```text
BM25 + Vector Search
        ↓
EnsembleRetriever
        ↓
Compression 또는 reranker
        ↓
LLM
```

### 긴 기술 문서 RAG

```text
ParentDocumentRetriever
        ↓
관련 parent 문서
        ↓
필요 시 compression
        ↓
LLM
```

### Metadata가 중요한 검색

```text
SelfQueryRetriever
        ↓
의미 검색 + 구조화 필터
        ↓
애플리케이션 권한 필터 강제 적용
        ↓
LLM
```

## 공식 참고 문서

- [LangChain Retriever integrations](https://docs.langchain.com/oss/python/integrations/retrievers)
- [ContextualCompressionRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/contextual_compression/ContextualCompressionRetriever)
- [EnsembleRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever)
- [ParentDocumentRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever)
- [MultiQueryRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever)
- [MultiVectorRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_vector/MultiVectorRetriever)
- [Self-query retriever](https://reference.langchain.com/python/langchain-classic/retrievers/self_query)
- [TimeWeightedVectorStoreRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/time_weighted_retriever/TimeWeightedVectorStoreRetriever)
