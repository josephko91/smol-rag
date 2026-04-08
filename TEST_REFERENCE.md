# Quick Test Reference Card

## Key Test Questions (Built from Cloud Science Papers in `/docs/`)

Use this to quickly validate RAG is working and retrieving the right documents.

---

## ✓ Quick 1-Minute Test

Run this sequence to quickly verify all 6 fixes are working:

```bash
python3 << 'EOF'
from agent import ResearchAgent

agent = ResearchAgent()

# Q1: Basic retrieval
print("Q1: What are cirrus clouds?")
a1 = agent.answer("What are cirrus clouds?")
print(f"A1: {a1[:200]}...\n")

# Q2: MEMORY TEST - should reference Q1
print("Q2: Based on what you just said, why do they matter for climate?")
a2 = agent.answer("Based on what you just said, why do they matter for climate?")
print(f"A2: {a2[:200]}...\n")

# Check if memory worked (Q2 should reference Q1)
if "you just said" in a2.lower() or "previously mentioned" in a2.lower() or "discussed" in a2.lower():
    print("✓ PASS: Memory working - agent referenced previous answer")
else:
    print("✗ FAIL: Memory not working - agent forgot previous context")

print(f"\nConversation history: {len(agent.conversation_history)} turns")
print(f"Conversation: {agent.conversation_history}")
EOF
```

**Expected output**:
- A1 talks about ice crystals, high altitude, role in climate
- A2 explicitly references or builds on A1
- "Conversation history: 2 turns" shown
- Memory test shows ✓ PASS

---

## Test Questions by Category

### Category 1: Basic Facts (Should Always Work)

| Question | Document(s) | Expected Answer Keywords |
|----------|-------------|-------------------------|
| "What are cirrus clouds?" | waliser_2009, zondlo_2000 | ice, crystals, altitude 6-12km, Earth's atmosphere |
| "What is the altitude of cirrus clouds?" | Multiple | upper troposphere, 6-12 km or 20,000-40,000 ft |
| "What is the main composition of cirrus clouds?" | Multiple | ice crystals or ice particles |

### Category 2: Climate Impact (Memory Test)

| Question 1 | Q1 Answer Should Say | Question 2 (Memory Test) | Expected: Q2 Should |
|-----------|---------------------|-------------------------|-------------------|
| "Explain how cirrus clouds affect Earth's radiation" | greenhouse effect, warming, infrared, trapping | "You just explained cirrus effects – why is understanding their ice structure important?" | **Reference Q1 answer**, explain connection |
| "How do clouds influence weather and climate?" | multiple roles, radiative, precipitation | "In that context, what challenges do models face?" | **Build on Q1**, discuss modeling difficulties |

**Key**: If Q2 uses words like "you said", "we discussed", "as mentioned" → Memory ✓ PASS

### Category 3: Technical Content

| Question | Document(s) | Expected Keywords |
|----------|-------------|-------------------|
| "What are the different shapes of ice particles?" | lawson_2019 | plates, columns, dendrites, aggregates, needles |
| "What is ice supersaturation?" | comstock_2008, zondlo_2000 | vapor pressure, critical radius, ice growth |
| "How do particle sizes affect cloud properties?" | Multiple | optical depth, reflectivity, scattering, absorption |

### Category 4: Model Challenges

| Question | Document(s) | Expected Keywords |
|----------|-------------|-------------------|
| "What are challenges in modeling cirrus clouds?" | waliser_2009, liu_2007 | parameterization, uncertainty, resolution, complexity |
| "How accurate are cloud parameterizations?" | Multiple | uncertain, limitations, improvements needed, observations |
| "What data improves cloud models?" | Multiple | satellite, aircraft, in situ, measurements |

---

## Verifying Each Fix

### 1. Debug Logging ✓
**Check for**: Logs should show retrieval details
```
[agent] INFO: Using RAG for question: What are cirrus clouds?
[retriever] INFO: Retrieved 12 documents
[agent] INFO: Doc 0: {...} | Preview: ...
```

### 2. Token-Aware Truncation ✓
**Check for**: Context length in tokens, not just characters
```
[agent] INFO: Formatted context: 2845 tokens, 11200 characters
```
(Should respect MAX_PROMPT_TOKENS limit)

### 3. Retrieval Instrumentation ✓
**Check for**: Batch details and retry information
```
[retriever] INFO: Embedding 60 texts in batches of 64
[retriever] DEBUG: Successfully embedded batch 1 (60 texts)
```

### 4. Lower Temperature ✓
**Check for**: Same questions producing similar answers
```python
a1 = agent.answer("What affects radiation?")
a2 = agent.answer("What affects radiation?")
# a1 and a2 should have >50% word overlap
```

### 5. Increased TOP_K ✓
**Check for**: 12 documents retrieved, not 8
```
[retriever] INFO: Retrieved 12 documents  ← should be 12, not 8
```

### 6. Conversation Memory ✓
**Check for**: History grows with each turn
```
[agent] INFO: Conversation history: 1 items
[agent] INFO: Conversation history: 2 items
[agent] INFO: Conversation history: 3 items
```

And follow-up Q2 should reference Q1:
```
Q2: "Based on what you just explained, why is X important?"
A2: Should explicitly reference Q1 or say "As I mentioned..." or "The points I raised..."
```

---

## Common Test Scenarios

### Scenario A: "Agent Forgets Facts" 🔧 Test

```python
# Q1: Establish a fact
q1 = "What is the typical altitude range of cirrus clouds?"
a1 = agent.answer(q1)
# A1 should say: "6 to 12 kilometers" or "20 to 40k feet"

# Q2: Ask follow-up 
q2 = "Given that altitude range you just mentioned, what atmospheric conditions would you expect?"
a2 = agent.answer(q2)

# Q3: Ask agent to recall
q3 = "Earlier you told me cirrus clouds are at [altitude]. Why is that altitude significant?"
a3 = agent.answer(q3)

# Success: A2 and A3 should reference the altitude fact from A1
```

**If FAILS**: Conversation memory not working, check:
- Does `agent.conversation_history` have items?
- Does prompt template include `{history}`?
- Is `temperature` set to 0.3 or lower?

### Scenario B: "Context Truncation" 🔧 Test

```python
# Ask a question that should retrieve many docs
long_answer = agent.answer("Comprehensively explain the role and effects of cirrus clouds on Earth's climate system including radiative processes")

# Check context length
if "1000 tokens" in str(agent.agent.conversation_history[-1][1]):  # rough check
    print("✓ Context properly sized (respecting token limits)")
else:
    print("Context may be truncated - check log for token count")
```

### Scenario C: "Consistency" 🔧 Test

```python
answers = []
for i in range(3):
    a = agent.answer("What is the greenhouse effect of cirrus clouds?")
    answers.append(a)

# Check if answers are similar
similar_1_2 = len(set(a1.split()) & set(a2.split())) / len(set(a1.split()) | set(a2.split()))
if similar_1_2 > 0.5:
    print(f"✓ Consistent answers ({similar_1_2:.1%} overlap)")
else:
    print(f"✗ Inconsistent ({similar_1_2:.1%} - check temperature)")
```

---

## Expected Documents to Retrieve

The papers in `/docs/` are the knowledge base:

1. **waliser_2009_jrg-atm** - Cloud ice climate challenges
2. **comstock_2008_jgr-atm** - Ice supersaturation and particle growth
3. **baker_1997_science** - Cloud microphysics and climate
4. **liou_1986_mwre** - Cirrus influence on weather/climate
5. **gasparini_2023_acp** - Tropical cirrus
6. **zondlo_2000_annu-rev** - Chemistry/microphysics of polar/cirrus
7. **wielicki_1995_bams** - Role of clouds in radiation
8. **lawson_2019_jgr-atm** - Ice particle shapes
9. **liu_2007_jclim** - Ice microphysics in CAM3

**Test retrieval directly**:
```bash
python inspect_database.py
```

---

## Passing Criteria

| Aspect | Metric | Pass Threshold |
|--------|--------|-----------------|
| **Basic Retrieval** | Non-empty answers | All 3 Q's answered |
| **Memory Retention** | Q2 references Q1 | ✓ Explicit reference |
| **Consistency** | Answer overlap | > 50% word match |
| **Context Size** | Token count | 500-4000 tokens |
| **Doc Retrieval** | Docs per query | 12 documents (TOP_K) |
| **Temperature** | Randomness | Low (0.3 or lower) |

---

## Running Full Test Suite

```bash
# Start Ollama if not running
# ollama serve &

# Run tests
python test_rag_fixes.py 2>&1 | tee test_run.log

# View results
cat test_results.json | python -m json.tool
```

**All 6 fixes should show**:
- ✓ Logging captured debug info
- ✓ Docs retrieved (12 per query)
- ✓ Memory grew (3+ turns in history)
- ✓ Consistency >50% on same questions
- ✓ Context properly sized
- ✓ Multi-turn follow-ups reference prior answers
