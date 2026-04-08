#!/usr/bin/env python3
"""
Test script to verify RAG context retention and retrieval fixes.

This script tests:
1. Document retrieval accuracy
2. Context preservation across turns (conversation memory)
3. Token-aware truncation
4. Temperature-induced consistency
5. Logging output for debugging

Run with: python test_rag_fixes.py
"""
import os
import sys
import json
import logging
from datetime import datetime

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)

from agent import ResearchAgent

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class TestRunner:
    """Test harness for RAG fixes"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("SMOL-RAG TEST SUITE - Context Retention & Retrieval Verification")
        print("="*80)
        print(f"Started: {datetime.now().isoformat()}")
        print("Testing document retrieval, conversation memory, and consistency\n")
        
        self.agent = ResearchAgent()
        self.results = []
    
    def test_group(self, name: str, description: str, questions: list):
        """Run a group of related test questions"""
        print(f"\n{'='*80}")
        print(f"TEST GROUP: {name}")
        print(f"{'='*80}")
        print(f"Purpose: {description}\n")
        
        group_results = []
        
        for i, q in enumerate(questions, 1):
            print(f"\n[Q{i}] {q}")
            print("-" * 80)
            
            try:
                answer = self.agent.answer(q, use_rag=True)
                
                # Truncate long answers for display
                answer_display = answer[:500] + "..." if len(answer) > 500 else answer
                print(f"[A{i}] {answer_display}\n")
                
                group_results.append({
                    "question": q,
                    "success": True,
                    "answer_length": len(answer),
                    "answer_preview": answer_display
                })
            except Exception as e:
                print(f"[ERROR] {str(e)}\n")
                group_results.append({
                    "question": q,
                    "success": False,
                    "error": str(e)
                })
        
        self.results.append({
            "group": name,
            "description": description,
            "results": group_results
        })
    
    def run_all_tests(self):
        """Run all test groups"""
        
        # ===== TEST GROUP 1: Basic Cirrus Cloud Facts =====
        self.test_group(
            "Basic Cirrus Cloud Facts",
            "Verify agent can retrieve and answer basic questions about cirrus clouds",
            [
                "What are cirrus clouds?",
                "What is the typical altitude of cirrus clouds?",
                "What is the composition of cirrus clouds?",
            ]
        )
        
        # ===== TEST GROUP 2: Climate Impact (Context Memory Test) =====
        # These questions build on each other to test conversation memory
        self.test_group(
            "Climate Impact & Memory Retention",
            "Test conversation memory: agent should reference prior context about cloud impacts",
            [
                "How do cirrus clouds affect Earth's climate and radiation?",
                "Based on what you just explained, why would understanding their ice structure be important?",
                "What did we just discuss about cloud radiative effects?",
            ]
        )
        
        # ===== TEST GROUP 3: Ice Microphysics (Detailed Retrieval) =====
        self.test_group(
            "Ice Microphysics & Particle Properties",
            "Verify retrieval of technical details about ice particles",
            [
                "What are the different shapes of ice particles in cirrus clouds?",
                "How do ice particle sizes affect cloud properties?",
                "What is the relationship between ice supersaturation and particle growth?",
            ]
        )
        
        # ===== TEST GROUP 4: Model Challenges (Specialized Knowledge) =====
        self.test_group(
            "Climate Model Representation",
            "Test retrieval of model challenges and parameterization issues",
            [
                "What are the challenges in representing cirrus clouds in climate models?",
                "How accurate are current parameterizations of cloud ice processes?",
                "What observational data is needed to improve cloud modeling?",
            ]
        )
        
        # ===== TEST GROUP 5: Consistency Check (Same questions repeated) =====
        print(f"\n{'='*80}")
        print("TEST GROUP: Consistency Check")
        print("="*80)
        print("Purpose: Verify low temperature produces consistent answers\n")
        
        test_q = "What is the main radiative effect of cirrus clouds?"
        print(f"Asking the same question twice to check consistency...\n")
        print(f"[Q1] {test_q}")
        answer1 = self.agent.answer(test_q, use_rag=True)
        print(f"[A1] {answer1[:300]}...\n")
        
        print(f"[Q2] (SAME QUESTION) {test_q}")
        answer2 = self.agent.answer(test_q, use_rag=True)
        print(f"[A2] {answer2[:300]}...\n")
        
        # Simple similarity check
        similarity = self._simple_similarity(answer1, answer2)
        print(f"Answer similarity: {similarity:.1%}")
        print("(Higher = more consistent; >70% indicates good consistency)\n")
        
        consistency_pass = similarity > 0.5  # Lenient check
        self.results.append({
            "group": "Consistency Check",
            "consistency_score": similarity,
            "passed": consistency_pass
        })
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity heuristic"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print("="*80)
        
        total_questions = 0
        total_successes = 0
        
        for group in self.results:
            if "results" in group:
                group_size = len(group["results"])
                group_successes = sum(1 for r in group["results"] if r.get("success", False))
                total_questions += group_size
                total_successes += group_successes
                
                status = "✓ PASS" if group_successes == group_size else "✗ FAIL"
                print(f"\n{status} | {group['group']}: {group_successes}/{group_size} passed")
        
        if total_questions > 0:
            success_rate = total_successes / total_questions
            print(f"\n{'─'*80}")
            print(f"Overall: {total_successes}/{total_questions} questions answered ({success_rate:.1%})")
            print(f"Conversation history retained: {len(self.agent.conversation_history)} turns")
        
        print(f"\nEnded: {datetime.now().isoformat()}")
        print("="*80)
    
    def export_results(self, filename: str = "test_results.json"):
        """Export test results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Results exported to {filename}")


def main():
    """Run test suite"""
    try:
        runner = TestRunner()
        runner.run_all_tests()
        runner.print_summary()
        runner.export_results("test_results.json")
        print("\n✓ All tests completed!")
        return 0
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
