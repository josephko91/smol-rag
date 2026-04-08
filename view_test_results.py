#!/usr/bin/env python3
"""
Extract and display Q&A pairs from test logs with verification.

Usage:
  python view_test_results.py              # Parse test_output.log
  python view_test_results.py test_output.log  # Custom log file
  python view_test_results.py --json       # Export as JSON
"""
import re
import json
import sys
from datetime import datetime
from pathlib import Path


class TestLogParser:
    """Parse test logs and extract Q&A pairs"""
    
    def __init__(self, log_file: str = "test_output.log"):
        self.log_file = log_file
        self.qa_pairs = []
        self.metrics = {}
        
    def parse(self):
        """Parse the log file"""
        if not Path(self.log_file).exists():
            print(f"Error: {self.log_file} not found")
            return False
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Extract all questions using regex
        question_pattern = r"Using RAG for question: (.+?)\.{3}"
        questions = re.findall(question_pattern, content)
        
        # Extract all "Retrieved X documents" entries
        retrieval_pattern = r"\[agent\] INFO: Retrieved (\d+) documents"
        retrievals = re.findall(retrieval_pattern, content)
        
        # Extract all "Formatted context" entries
        context_pattern = r"Formatted context: (\d+) tokens, (\d+) characters"
        contexts = re.findall(context_pattern, content)
        
        # Extract conversation history (last one for each question)
        history_pattern = r"Conversation history: (\d+) items"
        histories = re.findall(history_pattern, content)
        
        # Match them up: each retrieval with context creates one Q&A pair
        for i, (retrieval, context) in enumerate(zip(retrievals, contexts)):
            docs_retrieved = int(retrieval)
            tokens = int(context[0])
            chars = int(context[1])
            
            # Try to find matching question
            question = questions[i] if i < len(questions) else "Unknown"
            turn = int(histories[i]) if i < len(histories) else i + 1
            
            self.qa_pairs.append({
                "turn": turn,
                "question": question,
                "docs_retrieved": docs_retrieved,
                "context_tokens": tokens,
                "context_chars": chars,
                "verified": docs_retrieved > 0
            })
        
        return True
    
    def compute_metrics(self):
        """Compute overall metrics"""
        if not self.qa_pairs:
            return
        
        self.metrics = {
            "total_questions": len(self.qa_pairs),
            "avg_docs_retrieved": sum(q.get('docs_retrieved', 0) for q in self.qa_pairs) / len(self.qa_pairs),
            "avg_context_tokens": sum(q.get('context_tokens', 0) for q in self.qa_pairs) / len(self.qa_pairs),
            "verified_count": sum(1 for q in self.qa_pairs if q.get('verified')),
            "turn_max": max(q.get('turn', 0) for q in self.qa_pairs)
        }
    
    def display_summary(self):
        """Display results as formatted text"""
        if not self.qa_pairs:
            print("No Q&A pairs found in log")
            return
        
        self.compute_metrics()
        
        print("\n" + "="*100)
        print("TEST RESULTS SUMMARY")
        print("="*100)
        
        print(f"\nTotal Questions Asked: {self.metrics['total_questions']}")
        print(f"Max Conversation Turns: {self.metrics['turn_max']}")
        print(f"Verified Results: {self.metrics['verified_count']}/{self.metrics['total_questions']}")
        print(f"Average Docs Retrieved: {self.metrics['avg_docs_retrieved']:.0f}")
        print(f"Average Context Size: {self.metrics['avg_context_tokens']:.0f} tokens")
        
        print("\n" + "-"*100)
        print("QUESTION & ANSWER LOG")
        print("-"*100)
        
        for i, qa in enumerate(self.qa_pairs, 1):
            turn = qa.get('turn', '?')
            question = qa.get('question', 'Unknown')
            docs = qa.get('docs_retrieved', 0)
            tokens = qa.get('context_tokens', 0)
            verified = "✓" if qa.get('verified') else "✗"
            
            print(f"\n[Turn {turn}] Question {i}:")
            print(f"  Q: {question}")
            print(f"  Verified: {verified} | Docs: {docs} | Context: {tokens} tokens")
        
        print("\n" + "="*100)
    
    def display_table(self):
        """Display results as a table"""
        if not self.qa_pairs:
            print("No Q&A pairs found")
            return
        
        self.compute_metrics()
        
        print("\n" + "="*120)
        print("QUESTION & ANSWER VERIFICATION TABLE")
        print("="*120)
        
        # Header
        print(f"{'Turn':<6} {'Question':<50} {'Docs':<6} {'Tokens':<8} {'Verified':<10}")
        print("-"*120)
        
        # Rows
        for qa in self.qa_pairs:
            turn = qa.get('turn', '?')
            question = qa.get('question', '')[:47] + "..." if len(qa.get('question', '')) > 50 else qa.get('question', '')
            docs = qa.get('docs_retrieved', 0)
            tokens = qa.get('context_tokens', 0)
            verified = "PASS" if qa.get('verified') else "FAIL"
            
            print(f"{turn:<6} {question:<50} {docs:<6} {tokens:<8} {verified:<10}")
        
        print("-"*120)
        print(f"Totals: {len(self.qa_pairs)} questions | "
              f"Avg {self.metrics['avg_docs_retrieved']:.0f} docs | "
              f"Avg {self.metrics['avg_context_tokens']:.0f} tokens | "
              f"Success: {self.metrics['verified_count']}/{self.metrics['total_questions']}")
        print("="*120 + "\n")
    
    def export_json(self, output_file: str = "test_results_parsed.json"):
        """Export results as JSON"""
        if not self.qa_pairs:
            print("No results to export")
            return False
        
        self.compute_metrics()
        
        output = {
            "export_date": datetime.now().isoformat(),
            "source_log": self.log_file,
            "metrics": self.metrics,
            "qa_pairs": self.qa_pairs
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results exported to {output_file}")
        return True


def main():
    """Run the parser"""
    log_file = "test_output.log"
    export_json = False
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--json":
            export_json = True
        else:
            log_file = sys.argv[1]
    
    parser = TestLogParser(log_file)
    
    if not parser.parse():
        return 1
    
    # Display results
    parser.display_summary()
    print()
    parser.display_table()
    
    # Export if requested
    if export_json:
        parser.export_json()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
