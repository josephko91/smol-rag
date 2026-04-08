# How to View Test Results

There are now **three ways** to see what questions were asked, the answers, and verification:

## 1. **Quick Text Summary** (Recommended)

```bash
python view_test_results.py
```

Shows:
- All questions asked in conversation order
- Verification status (✓ PASS or ✗ FAIL)
- Number of documents retrieved per question
- Context size in tokens
- Summary metrics (avg docs, avg tokens, success rate)

Example output:
```
[Turn 1] Question 1:
  Q: What are cirrus clouds?
  Verified: ✓ | Docs: 12 | Context: 3701 tokens

[Turn 2] Question 2:
  Q: What is the typical altitude of cirrus clouds?
  Verified: ✓ | Docs: 12 | Context: 3701 tokens
```

## 2. **Table Format** (For spreadsheet-style review)

Same command as above, includes both summary log and detailed table:
```
Turn   Question                                           Docs   Tokens   Verified  
1      What are cirrus clouds?                            12     3701     PASS      
2      What is the typical altitude of cirrus clouds?     12     3701     PASS      
3      What is the composition of cirrus clouds?          12     3644     PASS      
```

## 3. **JSON Export** (For programmatic processing)

```bash
python view_test_results.py --json
```

Creates `test_results_parsed.json` with:
- All Q&A pairs
- Metrics (averages, totals)
- Token counts
- Document retrieval details
- Verification status

Usage:
```bash
# View JSON
cat test_results_parsed.json | python -m json.tool

# Parse with your own tools
python -c "import json; data=json.load(open('test_results_parsed.json')); print(data['metrics'])"
```

---

## Workflow

1. **Run tests** (produces `test_output.log`):
   ```bash
   timeout 300 python test_rag_fixes.py > test_output.log 2>&1
   ```

2. **View results** immediately:
   ```bash
   python view_test_results.py
   ```

3. **Export for records**:
   ```bash
   python view_test_results.py --json
   ```

---

## What Each Column Means

| Column | Meaning | Details |
|--------|---------|---------|
| **Turn** | Conversation turn number | 1-5 (memory maintains last 5 turns) |
| **Question** | The question asked | Extracted from logs |
| **Docs** | Documents retrieved | Should be 12 (TOP_K=12) |
| **Tokens** | Context size in tokens | Should be ~3500-4000 (respects MAX_PROMPT_TOKENS) |
| **Verified** | PASS/FAIL | PASS if docs > 0, indicates retrieval worked |

---

## Example Full Output

```
====================================================================================================
TEST RESULTS SUMMARY
====================================================================================================

Total Questions Asked: 6
Max Conversation Turns: 6
Verified Results: 6/6
Average Docs Retrieved: 12
Average Context Size: 3593 tokens

----------------------------------------------------------------------------------------------------
QUESTION & ANSWER LOG
----------------------------------------------------------------------------------------------------

[Turn 1] Question 1:
  Q: What are cirrus clouds?
  Verified: ✓ | Docs: 12 | Context: 3701 tokens

[Turn 2] Question 2:
  Q: What is the typical altitude of cirrus clouds?
  Verified: ✓ | Docs: 12 | Context: 3701 tokens

[Turn 3] Question 3:
  Q: What is the composition of cirrus clouds?
  Verified: ✓ | Docs: 12 | Context: 3644 tokens

[Turn 4] Question 4:
  Q: How do cirrus clouds affect Earth's climate and radiation?
  Verified: ✓ | Docs: 12 | Context: 3644 tokens

[Turn 5] Question 5:
  Q: Based on what you just explained, why would understanding their ice structure be important?
  Verified: ✓ | Docs: 12 | Context: 3433 tokens

[Turn 6] Question 6:
  Q: What did we just discuss about cloud radiative effects?
  Verified: ✓ | Docs: 12 | Context: 3433 tokens

========================================================================================================================
QUESTION & ANSWER VERIFICATION TABLE
========================================================================================================================
Turn   Question                                           Docs   Tokens   Verified  
1      What are cirrus clouds?                            12     3701     PASS      
2      What is the typical altitude of cirrus clouds?     12     3701     PASS      
3      What is the composition of cirrus clouds?          12     3644     PASS      
4      How do cirrus clouds affect Earth's climate...     12     3644     PASS      
5      Based on what you just explained, why would...     12     3433     PASS      
6      What did we just discuss about cloud radiative...  12     3433     PASS      
========================================================================================================================

Totals: 6 questions | Avg 12 docs | Avg 3593 tokens | Success: 6/6
```

---

## Using Custom Log File

If you want to parse a different log file:

```bash
python view_test_results.py my_test_run.log
python view_test_results.py my_test_run.log --json
```

---

## Key Observations to Look For

### ✓ Signs Everything is Working

| Check | Expected | What it means |
|-------|----------|---------------|
| **Verified** | All PASS | Documents retrieved successfully |
| **Docs** | 12 each | TOP_K increase working |
| **Tokens** | 3500-4000 | Token truncation respecting limits |
| **Turn growth** | 1→2→3→4→5 | Conversation memory working |
| **Avg Context** | ~3600 | Proper context sizing |

### ✗ Warning Signs

| Check | If This Happens | Likely Cause |
|-------|-----------------|--------------|
| **Verified: FAIL** | Some say FAIL | Retrieval not working |
| **Docs: < 12** | Fewer docs retrieved | Embedding or Chroma issue |
| **Tokens: very low** | <500 tokens | Docs being truncated |
| **Turn: stuck at 1** | Never grows | Memory not tracking turns |

---

## Next Steps

1. Run the full test suite: `timeout 300 python test_rag_fixes.py > test_output.log 2>&1`
2. View results: `python view_test_results.py`
3. Export: `python view_test_results.py --json`
4. Verify all metrics match expectations above

The `view_test_results.py` script makes it easy to verify all 6 fixes are working by showing exactly which questions were asked and confirming documents were retrieved properly.
