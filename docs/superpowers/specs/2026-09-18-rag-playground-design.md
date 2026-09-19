# RAG 청킹/검색 비교 플레이그라운드 — 설계 스펙

- 날짜: 2026-09-18
- 상태: 승인된 설계 (구현 전) — 2026-09-18 2차 브레인스토밍 반영
- 작업 명칭: **chunklab** (확정. 포지셔닝이 "청킹 벤치마크"로 좁아져 ragmatrix/retrievelab 탈락. 도메인·PyPI 가용성만 M3 전 확인)

## 1. 제품 정의

**"Benchmark your chunking before you ship it."**
문서를 넣으면 테스트 질문을 자동 생성하고, 청킹 전략(주축) × 임베딩 모델 × 검색 파라미터(부가축) 조합을 매트릭스로 실행해 결과를 나란히 시각 비교하는 로컬 우선 개발자 도구.

포지셔닝을 "RAG A/B 테스트" 전반이 아니라 **청킹 벤치마크**로 좁힌 이유: 전자는 Ragas/Langfuse와 검색어 경쟁, 후자는 공백이며 Chroma 청킹 논문으로 수요가 검증됨.

### 타깃 사용자
- RAG 파이프라인을 구축 중인 개발자 (글로벌, 영어 UI/문서)
- 현재 청킹/임베딩 선택을 감이나 스프레드시트 수동 비교로 하고 있는 사람

### 핵심 가치 제안
1. 조합 비교를 수동 스크립트 대신 한 화면에서 (시각적 비교)
2. **문서와 API 키가 절대 외부 서버로 나가지 않음** (로컬 실행) — 핵심 세일즈 포인트
3. 설정 조합별 정량 메트릭(hit@k, MRR, NDCG, IoU) + 정성 확인(청크 프리뷰)을 동시에
4. 설치 후 5분 안에 첫 비교 결과 — 테스트 질문 자동 생성으로 라벨링 노동 제거

### 경쟁 지형과 포지셔닝
- Langfuse/LangSmith: 프로덕션 관측/트레이싱 — 겹치지 않음 (개발 단계 도구로 포지셔닝)
- Ragas/DeepEval: 평가 라이브러리(코드) — 우리는 "보이는 도구"(UI)로 차별화
- Chroma `chunking_evaluation`: 청킹 비교 논문 + 연구 리포. IoU/recall 메트릭 정의의 원조. UI 없고 유지보수 안 됨 → 우리는 그 방법론을 **제품화** ("논문의 방법론을 클릭으로")
- Chonkie: 청커 라이브러리. 비교·평가 안 함 → 경쟁이 아니라 **플러그인 대상** (Chonkie 청커를 감싸면 청커 수가 공짜로 늘어남)
- ChunkViz: 청킹 시각화 웹 데모. 검색·메트릭 없음. "chunking visualizer" 검색어는 겹침 → SEO 타깃
- 빈틈: 청킹 설정을 정량 벤치마크하고 시각 비교하는 도구는 공백

## 2. MVP 범위

### 포함 (v1)
| 항목 | 내용 |
|---|---|
| 문서 입력 | md / txt / pdf 업로드 (pymupdf). 업로드 시 1회 파싱해 정규화 plain text로 저장, 이후 모든 청커·정답은 이 텍스트 기준 |
| 정답 모델 | **golden span** = `(doc_id, start, end)` 문자 offset 범위. 질문당 여러 span 허용. 청커 독립적 (2.1절) |
| 테스트 질문 생성 | ① 자동 합성(문서 샘플링 → LLM 질문 생성, 첫 화면 "Generate 20 questions") ② span 드래그 → LLM이 질문 생성 ③ 수동(span 드래그 + 질문 입력). 생성된 질문은 삭제·span 수정 가능 |
| 청커 3종 | recursive, sentence-window, markdown-aware (semantic은 v1.1) |
| 임베딩 3계열 | OpenAI, Gemini, sentence-transformers(로컬, `chunklab[local]` extra) |
| 파라미터 축 | chunk size, overlap, top-k, hybrid(BM25+dense) on/off |
| 메트릭 | hit@k, MRR, NDCG, **IoU / precision** (2.1절) |
| 결과 화면 | 조합별 컬럼 나란히 비교 + 검색된 청크 원문 하이라이트 프리뷰 |
| 결과 export | 이긴 조합을 코드 스니펫으로 복사 (LangChain / LlamaIndex / 순수 Python 3탭) |
| CI 회귀 감지 | `chunklab run --config x.yaml --baseline prev.json --fail-below hit@5=0.8` — 무료 로컬 |
| 실행 방식 | `pip install chunklab` → `chunklab ui` → localhost:7860 |

### 2.1 정답 판정과 메트릭 정의
- **hit 판정**: 검색된 청크가 span의 X% 이상을 포함하면 hit (기본 X=50, 설정 가능). 기준을 청크가 아닌 span에 두어 큰 청크가 자동으로 이득 보지 않게 함
- **hit@k / MRR**: 위 hit 정의 사용
- **NDCG**: 청크별 gain = 해당 청크가 **새로 ** 덮는 정답 span 비율(앞 순위 청크가 이미 덮은 부분은 제외, span별 marginal coverage 합). IDCG = min(정답 span 수, k)개 슬롯에 gain 1.0. 겹치는 청크가 같은 정답을 반복 커버해도 점수가 늘지 않음 (NDCG ≤ 1)
- **IoU / precision**: 검색된 총 토큰 중 정답 토큰 비율 (Chroma 방식). 이게 없으면 chunk size 2048이 항상 이겨 도구가 "큰 청크 쓰세요"라는 뻔한 답만 냄
- 기각한 정답 모델: 기준 청킹의 청크 ID(기준 청커 편향), free-form 정답 텍스트 fuzzy match(판정 흐릿)

### 2.2 테스트 질문 생성이 v1 필수인 이유
hit@k/MRR/NDCG 전부 ground truth 필수. 수동 라벨링 30분은 Show HN 유입 사용자가 안 함 → 자동 생성 없으면 MVP는 정성 프리뷰 도구로 전락. 대신 semantic chunker를 v1.1로 밀어 총량 유지 (semantic은 유일하게 실험 전 임베딩을 추가 호출하는 청커라 캐시 설계·비용 모두 복잡).

### 제외 (v2 이후, YAGNI)
- semantic chunker (v1.1 — 플러그인 구조라 추가 용이)
- 실험 이력 diff 화면 (v1.1 — SQLite 이력은 v1부터 저장)
- Chonkie 청커 플러그인 래핑 (v1.1)
- 생성(answer) 품질 평가, LLM-as-judge
- 프로덕션 모니터링 / 트레이싱
- 외부 벡터DB 연동 (Pinecone, Weaviate 등)
- 리랭커 비교
- 문서 폴더 watch 자동 재실행

### 비교 화면 목업 (승인된 방향)
```
┌─ upload docs ──────────────────────┐
│ query: "what is the refund policy" │
├──────────┬──────────┬──────────────┤
│ recursive│ markdown │ sentence     │
│ 512/50   │ -aware   │ window       │
├──────────┼──────────┼──────────────┤
│ hit@5 ✓  │ hit@5 ✓  │ hit@5 ✗      │
│ 0.82     │ 0.91     │ 0.64         │
│ [chunk   │ [chunk   │ [chunk       │
│  preview]│ preview] │  preview]    │
└──────────┴──────────┴──────────────┘
```

## 3. 아키텍처

### 구성
- **단일 Python 패키지** (Python 3.11+, uv/ruff/ty — modern-python 스택)
- FastAPI 로컬 서버 + **HTMX + Jinja + 바닐라 JS** (React 기각 — 3.1절)
- 저장소: 로컬 SQLite (실험 설정·결과·이력)
- PDF 파싱: pymupdf
- 의존성: 기본 설치는 OpenAI/Gemini 임베딩만(수 MB). `chunklab[local]` extra로 sentence-transformers + torch(~2GB) 분리 — 기본에 넣으면 설치 5분, 첫인상 망함
- 벡터 검색: in-process numpy/FAISS — 실험 규모(문서 수십~수백 개)라 외부 벡터DB 불필요
- BM25: rank-bm25 또는 자체 구현 (hybrid 축용)
- 임베딩 API 키: 사용자 로컬 환경변수에서만 읽음

### 컴포넌트 경계
```
chunklab/
├── core/          # 청킹·임베딩·검색·메트릭 엔진 (UI 무관, 라이브러리로도 사용 가능)
│   ├── chunkers/    # 청커 3종(v1), 공통 인터페이스: chunk(doc) -> list[Chunk]
│   ├── embedders/   # 임베딩 3계열, 공통 인터페이스: embed(texts) -> ndarray
│   ├── retrieval/   # dense / bm25 / hybrid 검색
│   ├── metrics/     # hit@k, MRR, NDCG
│   └── runner/      # 조합 매트릭스 실행, 결과를 JSON으로
├── server/        # FastAPI: 업로드, 실행 트리거, 결과 조회 API
├── ui/            # Jinja 템플릿 + HTMX + 바닐라 JS (빌드 도구 없음)
└── cli/           # `chunklab ui`, `chunklab run --config x.yaml` (headless)
```

설계 원칙:
- core는 server 없이 단독 사용 가능 (CLI headless 실행 → JSON 출력). M1 산출물이자 CI 연동(유료 기능)의 기반
- 청커/임베더는 플러그인 인터페이스 — 추가가 쉬워야 커뮤니티 기여 유도 가능 (Chonkie 래핑도 이 경로)

### 3.1 UI 스택 선택 근거
| | React + FastAPI | Gradio/Streamlit | HTMX + FastAPI + Jinja |
|---|---|---|---|
| M2 예상 | 6~8주 (빌드 파이프라인, 정적 번들 패키징) | 2주 | 3~4주 |
| span 드래그 라벨링 | 가능 | 거의 불가 (커스텀 컴포넌트) | 가능 (Selection/Range API, 바닐라 JS ~100줄) |
| 배포 | npm 빌드 산출물을 wheel에 포함 | pip만 | pip만 |

정답 모델이 span이므로 텍스트 선택 → offset 변환 UI가 필수 → Gradio 탈락. 솔로 주 10~15h에 React 빌드 파이프라인은 과함. v1 화면은 3개(업로드/질문셋, 매트릭스 설정, 결과 비교)라 HTMX로 감당 가능. 복잡해지면 서버 API는 그대로 두고 프론트만 교체.

### 데이터 흐름
1. 문서 업로드 → 1회 파싱 → 정규화 plain text 로컬 저장 (원본 PDF는 재파싱 안 함. 파서마다 추출 결과가 달라 span offset이 깨지는 문제 차단)
2. 테스트 질문 생성 (자동 합성 / span→질문 / 수동) → `(question, [span...])` 저장
3. 실험 정의(조합 매트릭스) → runner가 조합별로 청킹→임베딩→인덱싱
4. 질문 실행 → 조합별 검색 결과 + span overlap 기반 메트릭 계산
5. 결과를 SQLite에 저장, UI가 컬럼 비교로 렌더
6. 임베딩 결과는 (문서 해시, 청커 설정, 모델) 키로 캐시 — 조합 간 중복 임베딩 호출 방지 (API 비용 절감)

### 에러 처리
- 임베딩 API 실패: 조합 단위 부분 실패 허용 — 실패한 컬럼만 에러 표시, 나머지 결과는 정상 렌더
- PDF 파싱 실패: 해당 문서만 스킵하고 경고
- 대용량 문서: MVP에서는 문서당 크기 상한(예: 10MB)으로 단순화

### 3.2 M2 로컬 UI 상세 (2026-09-19 확정)

**실행**: `chunklab ui [--port 7860] [--workspace DIR]` → uvicorn으로 FastAPI 기동, 브라우저 자동 오픈. 워크스페이스 = 프로젝트 디렉터리 하나. 그 안에 `chunklab.db`(SQLite), `docs/`(업로드 원본), `.chunklab-cache.db`(임베딩 캐시)가 생김. CLI의 `exp.yaml`/`questions.json`과 **같은 데이터 모델**을 쓰며 UI에서 export/import 가능 — CLI 사용자와 UI 사용자가 같은 파일로 협업.

**저장소(SQLite, 워크스페이스 로컬)**
- `documents(id TEXT PK, name, source, content_hash, text)` — 정규화 텍스트를 1회 저장. 이후 모든 span/청크는 이 텍스트 기준(§3 데이터 흐름 1)
- `questions(id TEXT PK, text, spans_json, created_at)`
- `runs(run_id TEXT PK, created_at, config_yaml, result_json, status, error)` — 이력. diff 화면은 v1.1
- 문서 원문·API 키는 DB에 저장하지 않음(원본 파일은 `docs/`에, 키는 환경변수)

**화면 3개 (HTMX + Jinja + 바닐라 JS, 빌드 도구 없음, htmx.min.js는 패키지에 동봉 — CDN 없음)**
1. **Documents & Questions** `/`
   - 업로드(md/txt/pdf, 다중), 문서 목록(이름·글자수·질문 수), 삭제
   - 문서 뷰어: 정규화 텍스트를 **단일 `<pre id="doc">`** 로 렌더 → 브라우저 Selection의 텍스트 노드 offset이 곧 문자열 offset (변환 로직 불필요). 드래그 → 툴바 "Add question" / "Generate question from selection"
   - "Auto-generate N questions" (LLM은 환경변수 키, 없으면 버튼에 안내)
   - 질문 목록: 클릭 시 뷰어에서 span 하이라이트, 질문 텍스트 인라인 수정, span 재지정(드래그 후 "Set span"), 삭제
   - Export `questions.json` / Import
2. **Experiment** `/experiment`
   - 청커별 체크박스 + 파라미터 그리드 입력(콤마 구분 값), 임베더 다중 선택(레지스트리 + 커스텀 spec 입력), top_k, hybrid, hit_threshold
   - "Run" → 서버 백그라운드 스레드에서 `run_experiment` 실행, HTMX 1초 폴링으로 진행률(`done N/M`) 표시. 완료 시 `/results/{run_id}`로 이동
   - Export `exp.yaml`
3. **Results** `/results/{run_id}`
   - 상단: 조합 × 메트릭 표(정렬 가능), **추천 조합 배지**(아래 규칙)
   - 질문 선택 → 조합별 컬럼에 검색된 청크 프리뷰. 원문 뷰어에서 정답 span(녹색)·검색 청크(노랑)·겹침(진한 녹색) 하이라이트
   - 조합 선택 → **코드 스니펫**(LangChain / LlamaIndex / 순수 Python 3탭, 복사 버튼)
   - 이력: `/runs` 에 과거 실행 목록(날짜·조합 수·최고 hit@k), 클릭으로 재열람

**추천 규칙** (`core/runner/recommend.py`, CLI `chunklab recommend result.json`과 UI 배지가 공유)
1. 에러 조합 제외
2. hit@k가 최고값의 0.05 이내인 조합만 남김
3. 그중 precision 최대
4. 동률이면 청크 수(n_chunks) 적은 쪽
5. 결과: `{combo_id, reason: "hit@5 1.00 (best), precision 0.24 (best among top-hit)"}`

**CLI 추가 (M2)**: `chunklab ui`, `chunklab recommend RESULT.json [--json]`, `chunklab snippet RESULT.json --combo ID --framework langchain|llamaindex|python`, `chunklab run --json`(표 대신 결과 JSON을 stdout으로 — Claude Code 스킬 등 에이전트 연동용)

**보안/프라이버시**: 서버는 `127.0.0.1` 바인딩 기본, 업로드 파일명은 stem만 사용(경로 조작 차단), 파일당 10MB 상한 유지.

**테스트 전략**: FastAPI `TestClient`로 라우트·폼·백그라운드 실행(스레드) 검증, Jinja 렌더 결과 문자열 단언. 드래그 JS는 순수 브라우저 코드라 pytest 대상 외 — 대신 "span 저장 API가 offset을 그대로 저장하고 뷰어가 같은 텍스트를 렌더한다"를 서버 측에서 검증.

### 클라우드 (2단계, MVP 이후)
- 결과 리포트 공유 링크, 실험 이력 저장, 팀 워크스페이스, CI 회귀 감지
- 로컬 결과 JSON만 push — 문서 원문은 로컬에 남김 (프라이버시 스토리 유지)
- MVP 런칭 반응 확인 후 착수. 스택 미정 (이 스펙 범위 밖)

## 4. 수익화 & GTM

| 구분 | 내용 | 가격 |
|---|---|---|
| OSS (MIT) | 로컬 도구 전체 + **CI 회귀 감지(headless CLI)** | 무료 — GitHub 스타 = 유입 채널 |
| Cloud 개인 | 공유 리포트, 실험 이력 대시보드 | $19/월 |
| Cloud 팀 | 워크스페이스, CI 결과 집계·이력 대시보드, 알림 | $49/월/팀 |

CI 회귀 감지를 무료 로컬로 푼 이유: 매 PR마다 실행되는 것이 최강 retention이고, "CI 결과를 어딘가에 모아 보고 싶다"는 자연스러운 다음 욕구가 클라우드 유료 가치가 됨. 유료로 잠그면 OSS 채택 자체가 안 됨.

### 로컬 retention 훅 (v1)
- 코드 스니펫 export: 결과가 실제 프로젝트로 넘어가야 "쓸모 있었다"로 기억됨
- CI 회귀 감지: `--baseline` + `--fail-below`로 GitHub Actions에서 재실행

- 유입: Show HN, r/LangChain, r/LocalLLaMA, "chunking strategy comparison" / "chunking visualizer" / "chunking benchmark" 블로그 SEO
- 영업 0시간 전제 (셀프서브)
- 반복 사용 동기 = 이력/회귀 감지 → 클라우드 유료 가치를 여기에 집중

### 성공 지표
- 3개월: GitHub 스타 500+
- 6개월: 클라우드 베타 유료 고객 10명
- 미달 시: 피벗 또는 중단 판단 시점

## 5. 실행 계획 (주 10~15h)

| 마일스톤 | 기간 | 산출물 |
|---|---|---|
| M1: 코어 엔진 | 4주 | 청커 3종 + 임베딩 3계열 + span 메트릭 4종 + 질문 자동 생성, CLI headless 실행(`--baseline`/`--fail-below` 포함) → JSON 출력 |
| M2: 로컬 UI | 4주 | 업로드/질문셋(span 드래그), 매트릭스 실행, 컬럼 비교 화면, 코드 스니펫 export |
| M3: 런칭 | 4주 | 다듬기, 문서(README/데모 GIF), PyPI 배포, Show HN |
| M4+: 클라우드 | 미정 | 런칭 반응 본 뒤 착수 |

## 6. 리스크

1. **무관심 (최대 리스크)**: "한 번 쓰고 끝" 도구가 되기 쉬움 → v1 로컬 훅(코드 export, CI 회귀 감지)으로 대응. 클라우드에만 retention을 두지 않음. 런칭 반응이 수익화 가능성의 1차 검증
2. **프레임워크 내장화**: LangChain/LlamaIndex가 유사 기능 내장 가능 → 속도와 UX로 선점, 프레임워크 중립 포지셔닝
3. **API 비용**: 임베딩·질문 생성 LLM 호출은 사용자 부담(BYO key) — 서비스 원가에 포함되지 않음. 캐시로 사용자 비용도 최소화
4. **합성 질문 품질**: 자동 생성 질문이 실제 사용자 질문과 다를 수 있음 → 생성 후 삭제·수정 UI 제공, 수동 라벨링 경로 병존

## 7. 미결정 사항

- 도메인 / PyPI 패키지명 가용성 (M3 전 확인)
- 클라우드 스택 (M4 착수 시 별도 스펙)
- 질문 자동 생성용 LLM: 임베딩 키와 같은 provider(OpenAI/Gemini) 재사용 전제, 모델·프롬프트는 M1에서 결정
- hit 판정 기본 임계값 X=50%의 적정성 — M1에서 Chroma 논문 수치와 대조
