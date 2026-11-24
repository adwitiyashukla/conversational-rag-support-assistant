# conversational-rag-support-assistant

Conversational RAG over support documentation. You ask a question, the service finds the passage in the docs that answers it, writes a short answer with citations, checks that every citation actually supports its sentence, and keeps the conversation so follow-up questions work. It runs on a CPU laptop with no API keys: SQLite FTS5 for BM25, a small ONNX embedding model for dense retrieval, and a local model through Ollama for generation.

The part I wanted to measure is the follow-up problem. In a support chat the second question rarely contains the words that matter. One dialog in the benchmark goes "My insurance ended so what should i do", then "Don't do that I'll get insurance", then "How long do I have". Searching for the last message on its own finds nothing useful. The service has five ways to turn a conversation turn into a search query, and they are measured against each other on MultiDoc2Dial, a benchmark of 661 validation dialogs grounded in 488 real DMV, Social Security, Veterans Affairs and Federal Student Aid web pages.

## The follow-up problem

Every agent turn in the validation dialogs is annotated with the document span it was grounded in. For each of the 4,427 agent turns that have a usable span, the service gets the conversation up to the user message before it and has to retrieve that span. A chunk counts as a hit when it covers at least half of the annotated span or half of the chunk, whichever is shorter. Recall@k is the share of turns with a hit in the top k, doc@5 counts hits on the right document, and ms is retrieval time per turn on my laptop (an Intel Core 5 120U, CPU only). All rows use hybrid retrieval with no reranker.

| Query | recall@1 | recall@5 | recall@10 | doc@5 | MRR@10 | ms per turn |
| --- | --- | --- | --- | --- | --- | --- |
| last message only | 22.0 | 44.5 | 53.0 | 51.0 | 0.316 | 19 |
| window of 3 turns | 27.5 | 55.7 | 66.6 | 73.8 | 0.396 | 34 |
| whole history | 26.7 | 54.8 | 65.5 | 73.8 | 0.387 | 60 |
| window + last message fused | 29.7 | 61.4 | 73.4 | 73.6 | 0.431 | 67 |

Searching for the last message alone misses more than half of the gold passages at k=5. A window of the last three turns adds 11 points. The whole history does no better than the window and costs twice as much. The best setting retrieves twice, once with the window and once with the last message on its own, and merges the four ranked lists with reciprocal rank fusion. A message like "Don't do that I'll get insurance" still gets its own list, while the window supplies the topic. That is 61.4 recall@5, 17 points above the last message alone, and it is the default.

The obvious alternative is to ask the model to rewrite the follow-up into a standalone question. I ran that on a fixed random sample of 400 turns, with the other strategies on the same sample for comparison:

| Query | recall@1 | recall@5 | recall@10 | doc@5 | MRR@10 | ms per turn |
| --- | --- | --- | --- | --- | --- | --- |
| last message only | 23.5 | 45.0 | 52.8 | 50.5 | 0.329 | 21 |
| window of 3 turns | 29.0 | 56.0 | 66.8 | 75.0 | 0.403 | 35 |
| whole history | 27.0 | 54.0 | 65.8 | 76.0 | 0.387 | 58 |
| window + last message fused | 32.8 | 63.7 | 74.8 | 74.2 | 0.455 | 46 |
| LLM rewrite (qwen2.5:3b) | 31.5 | 61.3 | 68.2 | 74.5 | 0.445 | 127 |

The rewrite lands 2.5 points behind fusion at k=5 and 6.5 points behind at k=10, and the rewrite call itself took 4.6 s per turn on top of the retrieval time in the table. Fusion gets the same result from string concatenation. The rewrite strategy is still in the code because a bigger model behind the same endpoint may change this picture, but it is not the default.

Recall@5 for the fused query by domain: Student Aid 68.3, DMV 67.7, VA 59.2, SSA 52.8.

## BM25, dense, or both

Same 400 turn sample, window query, no reranker:

| Retriever | recall@1 | recall@5 | recall@10 | doc@5 | MRR@10 | ms per turn |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 only | 29.5 | 54.5 | 66.5 | 74.5 | 0.409 | 34 |
| dense only | 23.2 | 46.8 | 58.0 | 67.8 | 0.338 | 29 |
| hybrid (RRF) | 29.0 | 56.0 | 66.8 | 75.0 | 0.403 | 35 |

BM25 carries most of the weight on this corpus. My guess is that exact terms like plate surrender and lapse letter matter more here than paraphrase, and FTS5 matches them with no model at all. The dense leg adds 1.5 points at k=5 on top of BM25, and on its own it is 8 points behind. Hybrid stays the default because the dense leg is cheap and it is the part that catches paraphrased questions.

## Does a reranker help

The cross-encoder is Xenova/ms-marco-MiniLM-L-6-v2 through ONNX. It scores the top 20 fused candidates and reorders them. The query it scores against matters, so I tried the latest message and the full window text:

| Setup | recall@1 | recall@5 | recall@10 | doc@5 | MRR@10 | ms per turn |
| --- | --- | --- | --- | --- | --- | --- |
| last message only, no rerank | 23.5 | 45.0 | 52.8 | 50.5 | 0.329 | 21 |
| last message only, rerank by message | 26.5 | 48.8 | 54.2 | 54.2 | 0.365 | 1320 |
| window of 3 turns, no rerank | 29.0 | 56.0 | 66.8 | 75.0 | 0.403 | 35 |
| window of 3 turns, rerank by message | 30.8 | 56.0 | 66.8 | 68.0 | 0.427 | 1077 |
| window of 3 turns, rerank by window text | 26.8 | 50.7 | 60.5 | 71.8 | 0.374 | 1388 |
| LLM rewrite, no rerank | 31.5 | 61.3 | 68.2 | 74.5 | 0.445 | 127 |
| LLM rewrite, rerank by message | 28.5 | 58.5 | 64.8 | 69.2 | 0.412 | 1031 |

Reranking by the latest message moves the right passage up when it is already in the candidates (MRR 0.403 to 0.427 for the window) and helps the bare last message by almost 4 points, but it never adds recall@5 for the context aware queries and it hurts the rewritten ones. Scoring against the whole window text is worse than not reranking. A cross-encoder trained on single web search questions does not know what to do with three turns of conversation. At one second per turn on this CPU that is not worth it, so the reranker ships disabled and `enabled = true` in config.toml turns it back on.

With the reranker on, its top score is also the abstention signal. The service refuses to answer when the best candidate scores below `min_score`, and this is what that threshold buys on the 400 turn sample with the window query:

| min_score | turns answered | recall@5 when answered |
| --- | --- | --- |
| -8 | 95.0 | 56.8 |
| -4 | 77.5 | 64.5 |
| 0 | 57.8 | 73.2 |
| 4 | 35.5 | 78.2 |

## Answers and citations

For the generation side I took a fixed random sample of 120 agent turns and ran the whole pipeline: window query, hybrid retrieval, no reranker, qwen2.5:3b through Ollama, citations checked. The window query was the default when this run started and fusion became the default after the retrieval table above, so these numbers are for the weaker of the two queries; the retrieval table puts the gold passage in the top five 5.7 points more often with fusion. The reference answer for a turn is what the human agent said next, so token F1 against it is a loose measure: the agent often replies with a question of their own, or in twenty words where the model writes forty. The fine-tuned baselines in the MultiDoc2Dial repository report F1 between 30.9 and 33.7 on this split. A zero-shot 3B model with a chat prompt gets 20.9. The grounding numbers are the ones I care about.

| Measure | Value |
| --- | --- |
| turns | 120 |
| token F1 against the agent's reply | 20.9 |
| exact match | 0.0 |
| abstained | 15.8% |
| grounded, every cited sentence verified | 62.5% |
| grounded among answered turns | 74.3% |
| answers with at least one verified citation | 76 of 101 answered |
| a cited passage is a gold passage | 44.7% |
| gold passage in the five sources shown to the model | 55.0% |
| mean latency per turn | 38.5 s |
| p95 latency per turn | 56.3 s |

Almost all of the latency is the model. Generation averages 38.1 s on this CPU because every turn pushes about 1,300 prompt tokens through a 3B model, and the citation check takes 4 ms. Retrieval averaged 370 ms in this run against 34 ms in the retrieval table, because the model was competing for the same CPU between turns. On a machine with a GPU, or against a hosted endpoint, the same config.toml with a different `base_url` is the whole change.

Here is one turn from the sample, dialog 4a403522, turn 17. After fifteen turns about destroyed plates, police reports and registration, the user asks:

```
user: That's what I wanted to do. I also wanted to request new plates while I'm there, is that possible?
```

The window query carries the previous exchange about bringing the police report to a DMV office, retrieval lands on the plate surrender page, and the model answers in 31.9 s:

```
Yes, you can request new plates while you are at the DMV office. Bring the original plate or the
MV-78B / police report / statement, the completed Vehicle Registration / Title Application PDF,
and proof of your identity to the DMV office. [2] They will issue you a new registration and
replacement plates.
```

The agent's reference reply was "Yes, the DMV will issue you a new registration and replacement plates." The answer is marked grounded because its one citation checks out against source 2, and that source is the annotated gold span.

## Ingestion

`convrag ingest` is incremental. Every file is hashed, and a file whose hash matches the stored one is skipped without being read again. A changed file is deleted and re-inserted inside one transaction, so a crash in the middle leaves the previous version in place, and a file that disappeared from the folder is removed from the index. The timing on the benchmark corpus (488 documents, 2,942 chunks, embeddings on 8 ONNX threads):

| Run | Documents touched | Chunks embedded | Time |
| --- | --- | --- | --- |
| first sync | 488 added | 2,942 | 190.2 s |
| second sync, nothing changed | 0 | 0 | 124 ms |
| one file edited | 1 updated | 5 | 432 ms |

Embedding is the cost. Hashing 488 files and comparing them with the stored manifest takes 124 ms, so a scheduled re-sync of a documentation folder is free when nothing changed, and an edit to one page costs the five chunks of that page rather than the whole corpus.

## Serving

`convrag serve` runs the API on uvicorn with the models loaded once at startup. `convrag loadtest` sends 500 `/search` requests with real benchmark questions at a concurrency of 8 against a running server, which on the laptop gives 34 requests per second with p50 162 ms, p95 600 ms and p99 1218 ms, no errors. Each request does an FTS5 query, one ONNX embedding call and a matrix product over the 2,942 chunk vectors in a single worker process, and the tail comes from eight concurrent embedding calls sharing the same cores.

Every chat and search request produces a trace with the time spent in each stage. The last 500 traces are kept in SQLite and `/traces` returns them, and `/metrics` exposes request counts, abstentions by reason and per stage latency histograms in the Prometheus text format, so the service can sit behind a normal scrape job. The chat page at the root shows the queries that were searched, the sources with the cited ones highlighted, the grounded or abstained flag and the stage timings for every answer.

## How it works

Ingestion walks a folder, hashes every file, and only re-chunks and re-embeds files whose hash changed. Documents are split into sentence windows of about 180 words with one sentence of overlap, and every chunk keeps its character offsets in the original file. Each chunk is written to SQLite twice: once as a row with its embedding stored as a float32 blob, and once into an FTS5 table with the porter tokenizer for BM25. The document title is prefixed to the text that gets embedded and indexed, which is a cheap version of contextual chunk headers.

A chat turn goes through these stages, and each one is timed into a trace that comes back with the response and is kept in the database:

1. Query formulation. The configured strategy turns the conversation into one or two search queries: the last message alone, a window of the last three turns, the whole history, a fusion of window and last message, or an LLM rewrite into a standalone question.
2. Retrieval. BM25 over FTS5 and cosine similarity over the in-memory embedding matrix each return 50 candidates. The lists are fused with reciprocal rank fusion. With fusion queries there are four lists instead of two.
3. Reranking, if enabled. A cross-encoder scores the top candidates against the latest message and reorders them. If the best score is below a threshold the service abstains without calling the model.
4. Generation. The top five chunks go into the prompt as numbered sources with the recent conversation. The model is told to cite with [n] and to reply with a fixed token when the sources do not answer the question. Tokens are streamed to the client as they arrive.
5. Citation check. Every sentence that carries a citation is compared with the cited chunk by content word overlap, with a five letter prefix match so that "suspension" counts against "suspend". Citations that do not reach the overlap threshold are removed, and the answer is marked grounded only if every cited sentence kept at least one citation.
6. Persistence. The user message, the answer, its sources and the trace are stored under the session so the next turn can use them.

There is no LangChain or vector database in the dependency list. The retrieval and fusion code is under 200 lines, and it is easier to measure something you can read.

## Running it

Python 3.11 or newer. The embedding and reranking models download from Hugging Face on first use (about 150 MB together) into data/models. Generation talks to any OpenAI compatible endpoint, and the config points at a local Ollama by default.

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

On Linux or macOS use `source .venv/bin/activate` instead of the second line. Install Ollama from ollama.com and pull the default model:

```
ollama pull qwen2.5:3b
```

Index a folder of .md, .txt, .html or .pdf files and ask it something:

```
convrag ingest path\to\docs --collection mydocs
convrag search "how do I renew my license" --collection mydocs
convrag chat --collection mydocs
convrag serve
```

`convrag serve` starts the API on http://127.0.0.1:8000 with a small chat page at the root and OpenAPI docs at /docs. Re-running `convrag ingest` on the same folder only touches files that changed.

The benchmark run is three commands:

```
convrag data prepare
convrag ingest data\corpus\multidoc2dial --collection multidoc2dial
convrag eval all
```

`eval all` writes retrieval.json, generation.json and ingest.json into data/results. `convrag loadtest` hits a running server and writes loadtest.json next to them.

`docker compose up` builds the service and starts an Ollama container next to it. The compose file points the service at the container through `CONVRAG_LLM_BASE_URL`, which is how any value in config.toml can be overridden: `CONVRAG_<SECTION>_<KEY>`.

## API

| Method | Path | What it does |
| --- | --- | --- |
| POST | /chat | Answer a message. Send `session_id` to continue a conversation, `stream: true` for server-sent events |
| POST | /search | Retrieve chunks for a query, with `mode` (bm25, dense, hybrid) and `rerank` overrides |
| POST | /ingest | Sync a folder on the server into a collection |
| GET | /sessions/{id} | Messages of a session with their sources and traces |
| GET | /collections | Document and chunk counts per collection |
| GET | /traces | Recent request traces with per stage timings |
| GET | /metrics | Prometheus text format: request counts, stage latency histograms, abstentions by reason |
| GET | /health, /ready | Liveness and readiness, with the loaded model names |

A streamed chat returns `start`, `queries`, `sources`, `token` events and a final `done` event carrying the checked answer, the citation report and the trace. Setting `CONVRAG_API_KEY` makes every endpoint except /health and /ready require it as a bearer token.

## Configuration

Everything tunable lives in config.toml and is validated on startup: chunk sizes, the embedding and reranking models and their thread count, the candidate counts and RRF constant, the query strategy and window size, the LLM endpoint and model, the citation overlap threshold, the abstention threshold, and the evaluation paths. Secrets never go in the file. The LLM key is read from the environment variable named by `api_key_env`.

The embedding backend can be set to `hash` and the LLM provider to `fake`, which is how the tests run without downloading anything.

## Tests

93 tests cover chunk offsets, the SQLite store, incremental ingestion, the three retrieval modes and fusion, each query strategy, citation checking, the chat service including abstention and streaming, the API, config validation, metrics output and the evaluation harness on a miniature copy of the benchmark format. They use the hash embedder and the fake LLM, so CI runs them on Python 3.11 and 3.13 in about a minute.

## Data

MultiDoc2Dial by Feng, Patel, Wan and Joshi (EMNLP 2021). `convrag data prepare` downloads the archive from doc2dial.github.io, writes each document's text verbatim into data/corpus so that chunk offsets line up with the annotated grounding spans, and keeps the span table in gold.json for the evaluation. Nothing under data is committed.

## License

MIT
