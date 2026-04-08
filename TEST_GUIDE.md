# RAG Context Retention & Retrieval Tests

This directory contains test scripts to verify the fixes for the agent "forgetting" RAG context.

## Quick Start

### 1. Inspect What's in the Database

```bash
python inspect_database.py
```

This shows what documents are actually in the Chroma vector database and tests retrieval with sample queries. Good starting point to understand ingested content.

### 2. Run Full Test Suite

```bash
# Make sure Ollama is running with your model
# e.g., ollama serve &

python test_rag_fixes.py 2>&1 | tee test_run.log
```

This runs comprehensive tests across 5 groups:

1. **Basic Cirrus Cloud Facts** – Simple retrieval tests
2. **Climate Impact & Memory Retention** – Tests conversation memory with follow-ups
3. **Ice Microphysics** – Technical detail retrieval
4. **Climate Model Representation** – Specialized knowledge
5. **Consistency Check** – Verifies low temperature prevents contradictions

## What To Look For

### ✓ Successful Test Indicators

- **Q2 references Q1**: In "Climate Impact & Memory Retention" group, Q2 should reference what was discussed in Q1. This confirms conversation memory is working.
- **Consistent answers**: When the same question is asked twice in the "Consistency Check" group, answers should be similar (>50% word overlap). This confirms low temperature is working.
- **Retrieved context**: Agent should cite facts from documents, not hallucinate.
- **Logging shows docs**: Check logs for "Retrieved N documents" - should be retrieving 12 docs (TOP_K=12).

### ✗ Problem Indicators

- **Q2 forgets Q1 context**: If Q2 acts like Q1 never happened, conversation memory isn't working.
- **Completely different answers**: Same question producing totally different answers indicates high temperature or embedding issues.
- **"I don't have enough information"**: No documents retrieved; check embedding service is running.
- **Token counts very small**: If formatted context is <100 chars, docs might be truncated or retrieval failing.

## Details of Each Test Group

### Test Group 1: Basic Cirrus Cloud Facts
**Purpose**: Verify basic document retrieval works.

**Expected behavior**:
- Agent retrieves facts about cirrus clouds from papers
- Answers are grounded in document content
- Should reference altitude (~6-12 km), composition (ice), role in climate

### Test Group 2: Climate Impact & Memory Retention ⭐ KEY TEST
**Purpose**: Test conversation memory - the main fix for context "forgetting".

**Expected behavior**:
- Q1: Explains how cirrus clouds affect climate/radiation
- Q2: Agent remembers Q1 and explains why understanding ice structure relates to what was just discussed
- Q3: Agent can explicitly recall facts from Q1 and Q2

**This is the critical test** - if Q2 and Q3 forget Q1, conversation memory isn't working.

### Test Group 3: Technical Microphysics
**Purpose**: Verify retrieval of specialized technical content.

**Expected behavior**:
- Agent distinguishes between plate crystals, dendrites, spheres, etc.
- Discusses particle size effects on optical properties
- Explains ice supersaturation concepts

### Test Group 4: Model Representation
**Purpose**: Test retrieval of complex, domain-specific knowledge.

**Expected behavior**:
- Identifies parameterization challenges
- References uncertainty in model treatments
- May note need for better observations

### Test Group 5: Consistency Check
**Purpose**: Verify low temperature prevents contradictions.

**Expected behavior**:
- Asking same question twice produces similar (not identical) answers
- Key facts remain consistent
- Wording may vary slightly but content should align
- Similarity score should be >50%

## Debugging Failed Tests

### No documents retrieved?

1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check embedding model available: `ollama list` (should include `nomic-embed-text`)
3. Check Chroma database: `ls -lah chroma_data/`
4. Inspect database with: `python inspect_database.py`

### Agent keeps saying "I don't have information"?

1. Check `TOP_K` in config.py (should be 12)
2. Check temp in config.py (should be 0.3)
3. Try forcing RAG with: `use_rag=True` in code
4. Check logs for "Retrieved N documents" - should be >0

### Conversation memory not working?

1. Check conversation_history is populated: Look for "Conversation history: N items" in logs
2. Verify prompt template includes `{history}` - check agent.py line 62, 74
3. Test `clear_history()` - try `/reset-conversation` endpoint
4. Check that answers are being added to history: Look for "_add_to_history" in logs

### Inconsistent answers despite low temp?

1. Check MODEL_TEMPERATURE in config.py (should be 0.3)
2. Check MODEL_TEMPERATURE env var wasn't overridden: `echo $MODEL_TEMPERATURE`
3. Try lower: `export MODEL_TEMPERATURE=0.0` for fully deterministic
4. Check Ollama isn't cached differently - restart: `ollama serve --restart`

## Test Results Output

Tests create `test_results.json` with structured results:

```json
{
  "group": "Group Name",
  "results": [
    {
      "question": "What...",
      "success": true,
      "answer_length": 234,
      "answer_preview": "..."
    }
  ]
}
```

Parse with: `python -m json.tool test_results.json`

## Advanced Testing

### Test Only Retrieval (No LLM Generation)

```bash
python -c "
from retriever import Retriever
r = Retriever()
results = r.retrieve('cirrus cloud radiation')
for i, d in enumerate(results[:3]):
    print(f'{i}: {d[\"meta\"]} | {d[\"text\"][:100]}')
"
```

### Test Context Truncation

```bash
python -c "
from agent import format_docs, token_encoder
docs = [{'text': 'X' * 10000, 'meta': {}}]  # Very long doc
ctx = format_docs(docs)
print(f'Context length: {len(ctx)} chars')
if token_encoder:
    toks = len(token_encoder.encode(ctx))
    print(f'Token count: {toks} tokens')
"
```

### Manual Chat for Debugging

```bash
python3 -c "
from agent import ResearchAgent
agent = ResearchAgent()
print('Agent ready. Type your questions (Ctrl+C to quit):')
while True:
    q = input('\nQ: ')
    a = agent.answer(q)
    print(f'A: {a}')
    print(f'History: {len(agent.conversation_history)} turns')
"
```

## Log Interpretation

When running tests, watch logs for:

```
[agent] INFO: Using RAG for question: ...  ← RAG mode active
[retriever] INFO: Retrieved 12 documents   ← All 12 docs retrieved
[agent] INFO: Formatted context: 1200 tokens, 4800 characters
[agent] INFO: Doc 0: {...} | Preview: ...
[agent] INFO: Conversation history: 2 items  ← Memory growing
```

## Troubleshooting Summary

| Problem | Check | Fix |
|---------|-------|-----|
| No docs retrieved | Embedding service | `ollama pull nomic-embed-text` |
| Agent forgets Q1 in Q2 | Conversation memory | Verify `{history}` in prompts |
| Contradictory answers | Temperature | Lower `MODEL_TEMPERATURE` to 0.0-0.2 |
| Context cut off | Token counting | Verify tiktoken installed |
| Always says "no information" | RAG decision | Force `use_rag=True` |

---

**Key Reference**: All 6 fixes should work together:
1. ✓ Debug logging (see what's happening)
2. ✓ Token truncation (respect limits)
3. ✓ Retrieval logging (verify docs fetch)
4. ✓ Lower temperature (consistency)
5. ✓ Higher TOP_K (more docs)
6. ✓ Conversation memory (remember context)
