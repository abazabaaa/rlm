# RLM Example Tasks

This directory contains several examples demonstrating different capabilities of Recursive Language Models (RLMs).

## Available Examples

### 1. **main.py** - Needle in Haystack (Simple)
**Complexity:** ⭐ Basic
**What it tests:** Finding a specific value in massive unstructured text
**Context size:** ~5-7MB (1M lines)
**Key technique:** Direct regex search

```bash
python3 main.py
```

**Expected behavior:** The model will use regex to find "The magic number is X" in 1 million lines of random text.

---

### 2. **example_multi_document.py** - Multi-Document Reasoning
**Complexity:** ⭐⭐ Intermediate
**What it tests:** Cross-document semantic reasoning and filtering
**Context size:** ~5KB (small but interconnected)
**Key technique:** Multi-hop reasoning across related documents

```bash
python3 example_multi_document.py
```

**Challenge:** Find products from award-winning manufacturers with high ratings by connecting:
- Product → Manufacturer relationship
- Manufacturer → Awards data
- Product → Reviews → Average rating

**Expected behavior:** The model should:
1. Parse product, review, and manufacturer sections
2. Filter manufacturers by awards (2023+)
3. Calculate average ratings from reviews
4. Join the data to find qualifying products

---

### 3. **example_structured_data.py** - Large Dataset Analysis
**Complexity:** ⭐⭐⭐ Advanced
**What it tests:** Programmatic data analysis on structured JSON
**Context size:** ~500KB (5,000 log entries)
**Key technique:** Programmatic filtering and aggregation

```bash
python3 example_structured_data.py
```

**Challenge:** Analyze 5,000 user activity log entries to:
1. Find top 5 users with most failures
2. Calculate failure rates and patterns
3. Identify common failure types and pages

**Expected behavior:** The model should:
1. Parse the JSON data programmatically
2. Use Python to filter and aggregate (not regex)
3. Perform statistical analysis
4. Generate a structured report

**Similar to:** OOLONG benchmark mentioned in the blog post

---

### 4. **example_chunking_strategy.py** - Recursive Sub-LM Calls
**Complexity:** ⭐⭐⭐⭐ Expert
**What it tests:** Intelligent chunking with semantic analysis via sub-LMs
**Context size:** ~50KB (50 research papers)
**Key technique:** Chunking + recursive `llm_query()` calls

```bash
python3 example_chunking_strategy.py
```

**Challenge:** Find influential papers on specific topics from 50 papers using semantic search

**Expected behavior:** The model should:
1. Realize it needs semantic understanding (can't use simple regex)
2. Chunk papers into groups
3. Call `llm_query()` on each chunk to filter semantically
4. Aggregate results and synthesize findings

**This is the KEY RLM capability:** Using recursive sub-LLM calls for semantic analysis on large contexts.

---

## Understanding the Output

When you run these examples with `enable_logging=True`, you'll see:

- **Green boxes:** Query start and final results
- **Cyan text:** Model responses and reasoning
- **Code blocks:** Python code executed in the REPL
- **Execution results:** Output from code execution with timing

## Tips for Creating Your Own Tasks

1. **For simple pattern matching:** The model will use regex/string operations
2. **For semantic understanding:** The model should use `llm_query()` to delegate to sub-LLMs
3. **For large contexts:** Encourage chunking in your query
4. **For structured data:** Provide JSON for programmatic processing

## Performance Notes

From the blog post:
- RLM(GPT-5-mini) outperforms GPT-5 by 114% on OOLONG (132k tokens)
- RLM(GPT-5) achieves perfect performance on BrowseComp-Plus at 1000 documents (~10M+ tokens)
- Cost efficiency: Often cheaper than using the larger model directly

## Modifying Examples

Each example is self-contained. You can:
- Change `num_entries`, `num_papers`, etc. to scale context size
- Modify the queries to test different reasoning types
- Adjust `max_iterations` to give the model more/fewer steps
- Switch models (`gpt-5-nano`, `gpt-5-mini`, `gpt-5`)
