#!/usr/bin/env python3
"""
Example usage of the computational model
Demonstrates basic workflows
"""

from database_connector import DatabaseConnector
from vectorizer import EventVectorizer
from sequence_processor import SequenceProcessor
import json


def example_1_get_events():
    """Example: Get events from database"""
    print("Example 1: Getting events from database")
    
    db = DatabaseConnector()
    try:
        events = db.get_events(limit=10)
        print(f"Found {len(events)} events")
        for event in events[:3]:
            print(f"  - {event.get('type')} at {event.get('timestamp')}")
    finally:
        db.close()


def example_2_vectorize():
    """Example: Vectorize events"""
    print("\nExample 2: Vectorizing events")
    
    db = DatabaseConnector()
    vectorizer = EventVectorizer()
    
    try:
        events = db.get_events(limit=5)
        vectorizer.build_event_type_encoder(events)
        
        for event in events[:2]:
            vectorized = vectorizer.vectorize_event(event)
            print(f"  Event: {event.get('type')}")
            print(f"    Vector dim: {len(vectorized['combined_vector'])}")
            print(f"    Event type vector: {vectorized['event_type_vector']}")
    finally:
        db.close()


def example_3_calculate_cp():
    """Example: Calculate Context Precision"""
    print("\nExample 3: Calculating Context Precision")
    
    from scripts.calculate_cp import calculate_cp, extract_context_files
    
    db = DatabaseConnector()
    try:
        prompts = db.get_prompts(limit=5)
        
        for prompt in prompts[:2]:
            prompt_id = prompt.get('id')
            diff_files_data = db.get_entries_for_prompt(prompt_id, time_window_seconds=300)
            diff_files = [e.get('file_path') for e in diff_files_data if e.get('file_path')]
            
            cp_result = calculate_cp(prompt, diff_files)
            print(f"  Prompt {prompt_id}:")
            print(f"    CP: {cp_result['cp']:.2f}")
            print(f"    Context files: {cp_result['context_file_count']}")
            print(f"    Diff files: {cp_result['diff_file_count']}")
    finally:
        db.close()


def example_4_build_library():
    """Example: Build behavioral library"""
    print("\nExample 4: Building behavioral library")
    print("  (This may take a while with real data)")
    
    processor = SequenceProcessor()
    try:
        # Use a small workspace or limit sequences for demo
        library = processor.build_behavioral_library(workspace_path=None)
        
        print(f"  Total sequences: {library.get('total_sequences', 0)}")
        print(f"  Clusters found: {library.get('clusters_found', 0)}")
        
        # Save to file
        processor.save_library(library, output_path="example_library.json")
        print("  Saved to example_library.json")
    finally:
        processor.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_get_events,
            2: example_2_vectorize,
            3: example_3_calculate_cp,
            4: example_4_build_library
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1, 2, 3, 4")
    else:
        print("Computational Model - Example Usage")
        print("=" * 50)
        print("\nRun with example number:")
        print("  python example_usage.py 1  # Get events")
        print("  python example_usage.py 2  # Vectorize")
        print("  python example_usage.py 3  # Calculate CP")
        print("  python example_usage.py 4  # Build library")

