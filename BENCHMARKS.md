# RLM Benchmarks

This directory contains implementations of challenging benchmarks that demonstrate RLM's capabilities at scale.

## 📊 Available Benchmarks

### 1. OOLONG-Style Benchmark (`benchmark_oolong.py`)

**Based on:** [OOLONG: Evaluating Long Context Reasoning](https://openreview.net/forum?id=lrDr6dmXOX)

**What it tests:**
- Long-context reasoning over fine-grained information
- Semantic classification without explicit labels
- Aggregation across thousands of entries
- Filtering and counting with user-based constraints

**Task Description:**
Given 5,000 entries with format `Date: Oct 15, 2023 || User: 42 || Instance: Who is Marie Curie?`, the model must:
1. Parse all entries
2. Filter by specific user IDs
3. Infer implicit category labels ("entity", "description", "numeric", "temporal")
4. Count how many entries match a target category

**Why it's hard:**
- No explicit labels provided - must infer semantically
- Can't use simple keyword matching
- Requires processing entire dataset (no retrieval shortcuts)
- Original OOLONG: GPT-5 achieves <50% accuracy at 128K context

**Context size:** ~300KB (5,000 entries)

**Run it:**
```bash
python3 benchmark_oolong.py
```

**Expected behavior:**
- gpt-5-nano should parse and filter the data programmatically
- May use `llm_query()` to infer categories for chunks of questions
- Should aggregate results to get final count

---

### 2. BrowseComp-Plus Mini (`benchmark_browsecomp_mini.py`)

**Based on:** [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) ([GitHub](https://github.com/texttron/BrowseComp-Plus))

**What it tests:**
- Multi-hop reasoning across distributed documents
- Information synthesis from multiple sources
- Finding connections between manufacturers, products, ratings, and awards

**Task Description:**
Given 100 documents containing:
- Manufacturer information (awards, founding, headquarters)
- Product specifications (name, manufacturer, rating, year)
- Reviews and technical specs

Find products that match ALL criteria:
1. Made by an award-winning manufacturer
2. High rating (4.5+)
3. Released in recent years

**Why it's hard:**
- Answer requires information from multiple document types
- Must connect: Manufacturer → Awards → Products → Ratings
- Can't answer from a single document
- Full benchmark: 100K documents, ~10M+ tokens

**Context size:** ~35KB (100 documents)

**Run it:**
```bash
python3 benchmark_browsecomp_mini.py
```

**Expected behavior:**
- Should chunk documents by type
- Use `llm_query()` to extract structured information from chunks
- Join information across document types to find matches

---

## 📈 Performance Comparison

### Original Benchmark Results (from papers):

**OOLONG (132K tokens):**
- GPT-5: <50% accuracy
- RLM(GPT-5-mini): 114% improvement (relative)

**BrowseComp-Plus (1000 documents):**
- GPT-5 + BM25: 55.9% accuracy
- GPT-5 + Qwen3-Embedding: 70.1% accuracy
- RLM(GPT-5): Perfect accuracy at 1000 document scale

### These Mini Versions:

These are **scaled-down** versions designed to be runnable in a reasonable time:

| Benchmark | Original | Mini Version |
|-----------|----------|-------------|
| OOLONG | 132K tokens, thousands of entries | 300KB, 5,000 entries |
| BrowseComp-Plus | 100K docs, 10M+ tokens | 100 docs, 35KB |

The mini versions demonstrate the same **types of reasoning** but at a smaller scale.

---

## 🎯 Why These Benchmarks?

### vs. Simple Examples (needle-in-haystack)

**Simple examples** can be solved with regex:
```python
# Too easy - just search for pattern
answer = re.search(r'magic number is (\d+)', context)
```

**These benchmarks require:**
- ✅ Semantic understanding (can't regex category labels)
- ✅ Multi-document reasoning (must connect scattered info)
- ✅ Aggregation (counting across filtered subsets)
- ✅ Strategic chunking and recursive calls

### Real-World Applications

These benchmarks simulate:
- **OOLONG**: Customer support ticket classification, log analysis, survey data processing
- **BrowseComp-Plus**: Product research, competitive analysis, fact-checking across sources

---

## 🔍 Full Benchmarks

Want to try the full-scale benchmarks?

### OOLONG (Full)
**Status:** Dataset not yet publicly available (paper under review)
**Access:** Check [OpenReview page](https://openreview.net/forum?id=lrDr6dmXOX) for updates

### BrowseComp-Plus (Full)
**Status:** ✅ Available
**Access:** https://github.com/texttron/BrowseComp-Plus

**To download:**
```bash
cd browsecomp_data
pip install datasets huggingface_hub

# Download queries and answers
python scripts_build_index/decrypt_dataset.py \\
    --output data/browsecomp_plus_decrypted.jsonl \\
    --generate-tsv topics-qrels/queries.tsv

# Download full corpus (100K documents)
python -c "
from datasets import load_dataset
ds = load_dataset('Tevatron/browsecomp-plus-corpus', split='train')
ds.save_to_disk('data/corpus')
"
```

**Warning:** The full corpus is very large and requires significant compute to process.

---

## 💡 Understanding the Results

**Good signs:**
- Model uses `llm_query()` to delegate semantic tasks
- Chunks large datasets strategically
- Shows step-by-step reasoning in REPL

**Red flags:**
- Hallucinates answers without reading context
- Uses only regex/pattern matching (misses semantic requirements)
- Times out or runs out of iterations

**Debugging:**
- Check logs for `llm_query()` calls (should see recursive calls for semantic tasks)
- Verify the model reads the actual context (not making up answers)
- Look for programmatic aggregation in OOLONG (should use Python, not manual counting)

---

## 📚 References

1. **OOLONG Paper:** https://openreview.net/pdf?id=lrDr6dmXOX
2. **BrowseComp-Plus Paper:** https://arxiv.org/pdf/2508.06600
3. **BrowseComp-Plus GitHub:** https://github.com/texttron/BrowseComp-Plus
4. **RLM Blog Post:** https://alexzhang13.github.io/blog/2025/rlm/
