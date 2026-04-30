"""
Knowledge Base Example

Demonstrates how to create and use knowledge bases for improved entity extraction
with canonical names, aliases, and metadata.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from workplace_email_utils.entity_extraction.knowledge_base import KnowledgeBase, PersonInfo, OrganizationInfo
from workplace_email_utils.entity_extraction.extractors import extract_all_entities
from workplace_email_utils.ingest.email_parser import load_emails
from workplace_email_utils.entity_extraction.extractors import extract_entities_from_dataframe

# Example 1: Create a knowledge base programmatically
print("=" * 60)
print("Example 1: Creating Knowledge Base Programmatically")
print("=" * 60)

kb = KnowledgeBase()

# Add people with titles and aliases
kb.add_person(PersonInfo(
    canonical_name="John Smith",
    aliases=["John", "J. Smith", "Johnny", "J.S."],
    title="Senior Vice President",
    email="john.smith@company.com",
    organization="ABC Corporation"
))

kb.add_person(PersonInfo(
    canonical_name="Sarah Johnson",
    aliases=["Sarah", "S. Johnson", "Dr. Johnson", "Dr. Sarah Johnson"],
    title="Chief Technology Officer",
    email="sarah.j@company.com"
))

# Add organizations with aliases
kb.add_organization(OrganizationInfo(
    canonical_name="ABC Corporation",
    aliases=["ABC Corp", "ABC", "ABC Inc.", "ABC Industries"],
    domain="abc.com"
))

kb.add_organization(OrganizationInfo(
    canonical_name="XYZ Consulting",
    aliases=["XYZ", "XYZ LLC", "XYZ Consulting Group"],
    domain="xyz.com"
))

print(f"Knowledge base created:")
print(f"  Persons: {len(kb.persons)}")
print(f"  Organizations: {len(kb.organizations)}")

# Test entity extraction with knowledge base
test_text = """
Hi John,
I talked to Johnny yesterday about the ABC project.
Dr. Johnson mentioned that ABC Corp would be interested.
Please contact J.S. at john.smith@company.com.
"""

print("\nTesting entity extraction:")
result_without_kb = extract_all_entities(test_text)
print(f"\nWithout knowledge base:")
print(f"  Persons: {result_without_kb.persons}")

result_with_kb = extract_all_entities(test_text, knowledge_base=kb)
print(f"\nWith knowledge base:")
print(f"  Persons: {result_with_kb.persons} (canonical names)")
print(f"  Organizations: {result_with_kb.organizations} (canonical names)")

# Example 2: Save and load knowledge base
print("\n" + "=" * 60)
print("Example 2: Saving and Loading Knowledge Base")
print("=" * 60)

kb.save("knowledge_base.json")
print("✓ Saved knowledge base to knowledge_base.json")

kb_loaded = KnowledgeBase.load("knowledge_base.json")
print(f"✓ Loaded: {len(kb_loaded.persons)} persons, {len(kb_loaded.organizations)} organizations")

# Example 3: Load knowledge base from JSON file
print("\n" + "=" * 60)
print("Example 3: Creating Knowledge Base from JSON")
print("=" * 60)

# You can create a JSON file like this:
knowledge_base_json = {
    "persons": {
        "John Smith": {
            "canonical_name": "John Smith",
            "aliases": ["John", "J. Smith", "Johnny"],
            "title": "Senior VP",
            "email": "john.smith@company.com",
            "organization": "ABC Corporation",
            "metadata": {}
        },
        "Sarah Johnson": {
            "canonical_name": "Sarah Johnson",
            "aliases": ["Sarah", "Dr. Johnson"],
            "title": "CTO",
            "email": "sarah.j@company.com",
            "metadata": {}
        }
    },
    "organizations": {
        "ABC Corporation": {
            "canonical_name": "ABC Corporation",
            "aliases": ["ABC Corp", "ABC", "ABC Inc."],
            "domain": "abc.com",
            "metadata": {}
        }
    }
}

import json
with open("custom_kb.json", "w") as f:
    json.dump(knowledge_base_json, f, indent=2)

kb_from_json = KnowledgeBase.load("custom_kb.json")
print(f"✓ Loaded custom KB: {len(kb_from_json.persons)} persons")

# Example 4: Auto-generate knowledge base from emails
print("\n" + "=" * 60)
print("Example 4: Auto-generating Knowledge Base from Emails")
print("=" * 60)

from workplace_email_utils.entity_extraction.knowledge_base import create_knowledge_base_from_dataframe

df = load_emails('maildir', data_format='maildir', max_rows=100)
auto_kb = create_knowledge_base_from_dataframe(df)

print(f"✓ Auto-generated KB from emails:")
print(f"  Persons: {len(auto_kb.persons)}")
print(f"  Sample persons:")
for i, (name, person) in enumerate(list(auto_kb.persons.items())[:5]):
    print(f"    - {name} ({person.email})")

# Example 5: Using knowledge base in entity extraction pipeline
print("\n" + "=" * 60)
print("Example 5: Using Knowledge Base in Pipeline")
print("=" * 60)

# Merge auto-generated with custom
kb_combined = KnowledgeBase()
kb_combined.merge(auto_kb)
kb_combined.merge(kb_loaded)

print(f"✓ Combined KB: {len(kb_combined.persons)} persons")

# Extract entities using knowledge base
df_with_entities = extract_entities_from_dataframe(
    df.head(10),
    knowledge_base=kb_combined
)

print(f"✓ Extracted entities for {len(df_with_entities)} emails")
print(f"  Sample person entities: {df_with_entities['entities_persons'].iloc[0][:3] if len(df_with_entities['entities_persons'].iloc[0]) > 0 else 'None'}")

print("\n" + "=" * 60)
print("✓ All examples completed!")
print("=" * 60)

