#!/usr/bin/env python3
"""
Lightweight test to inspect what's actually in the Chroma vector database.

Run this first to see what documents are available and test retrieval directly.
"""
import os
import json

try:
    from retriever import Retriever
    
    print("\n" + "="*80)
    print("VECTOR DATABASE INSPECTOR")
    print("="*80)
    
    retriever = Retriever()
    
    # Test queries designed to match cirrus cloud papers
    test_queries = [
        "cirrus clouds",
        "ice particles",
        "cloud radiation",
        "altitude temperature",
        "climate impact",
        "microphysics",
    ]
    
    print("\nTesting retrieval with sample queries:\n")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        results = retriever.retrieve(query, k=2)
        
        if results:
            for i, doc in enumerate(results[:2], 1):
                meta = doc.get('meta', {})
                text = doc.get('text', '')
                
                print(f"  Result {i}:")
                print(f"    Source: {meta.get('source', 'unknown')}")
                print(f"    Text length: {len(text)} chars")
                print(f"    Preview: {text[:150].replace(chr(10), ' ')}...")
        else:
            print(f"  No results found")
    
    print("\n" + "="*80)
    print("✓ Database inspection complete")
    print("="*80 + "\n")

except ImportError as e:
    print(f"\n✗ Missing dependencies: {e}")
    print("\nTo run database inspection, install requirements:")
    print("  pip install -r requirements.txt")
    print("  Or: pip install chromadb langchain langchain-chroma langchain-ollama\n")
