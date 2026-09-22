파인콘(Pinecone) 벡터 데이터베이스의 기본 **CRUD(Create/Insert, Read/Query, Update, Delete)** 파이썬(Python) SDK 코드 예시.

---

### 사전 준비 및 연결

```python
from pinecone import Pinecone

# 1. API 키로 Client 초기화
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

# 2. 사용할 인덱스(Index) 연결
index_name = "my-index"
index = pc.Index(index_name)

```

---

### 1. Create / Insert (데이터 삽입)

`upsert` 파이프라인을 사용해 ID, 벡터 값(Embedding), 메타데이터(Metadata)를 삽입.

```python
# 삽입할 데이터 (ID, Vector, Metadata)
vectors_to_upsert = [
    {
        "id": "doc_1",
        "values": [0.1, 0.2, 0.3, ...], # embedding 차원 수에 맞춰 입력
        "metadata": {"category": "news", "title": "AI Trends"}
    },
    {
        "id": "doc_2",
        "values": [0.4, 0.5, 0.6, ...],
        "metadata": {"category": "blog", "title": "Pinecone Guide"}
    }
]

# 네임스페이스(선택사항) 및 데이터 저장
index.upsert(
    vectors=vectors_to_upsert,
    namespace="example-namespace"
)

```

---

### 2. Read / Query (유사도 검색 및 조회)

쿼리 벡터(Query Vector)와 유사한 벡터 상위 $k$개를 검색하거나 특정 ID 데이터 직접 조회(`fetch`) 가능.

#### A. 유사도 검색 (`query`)

```python
query_vector = [0.12, 0.22, 0.31, ...]

response = index.query(
    namespace="example-namespace",
    vector=query_vector,
    top_k=2,                     # 상위 N개 데이터 검색
    include_values=True,          # 벡터 값 포함 여부
    include_metadata=True,        # 메타데이터 포함 여부
    filter={"category": "news"}   # 메타데이터 기반 필터링 (선택)
)

print(response)

```

#### B. 특정 ID 직접 조회 (`fetch`)

```python
fetched_data = index.fetch(
    ids=["doc_1", "doc_2"],
    namespace="example-namespace"
)

print(fetched_data)

```

---

### 3. Update (데이터 수정)

`update` 메소드로 특정 ID의 벡터 값이나 메타데이터를 수정하거나, 기존 ID로 `upsert`를 재실행하여 덮어씀.

```python
# 특정 ID의 메타데이터 및 벡터 값 개별 수정
index.update(
    id="doc_1",
    values=[0.15, 0.25, 0.35, ...],
    set_metadata={"title": "Updated AI Trends"},
    namespace="example-namespace"
)

```

---

### 4. Delete (데이터 삭제)

ID 지정, 메타데이터 필터링, 또는 네임스페이스 전체 삭제 기능을 지원함.

```python
# 1. 특정 ID 삭제
index.delete(ids=["doc_1"], namespace="example-namespace")

# 2. 메타데이터 조건에 맞춰 삭제
index.delete(filter={"category": "blog"}, namespace="example-namespace")

# 3. 특정 네임스페이스 내부 데이터 전체 삭제
index.delete(delete_all=True, namespace="example-namespace")

```