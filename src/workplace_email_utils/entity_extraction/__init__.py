"""
Entity extraction and Named Entity Recognition (NER) module.

Extracts structured information from unstructured email text:
- People (names)
- Organizations (companies, departments)
- Locations (cities, countries, offices)
- Financial entities (amounts, contracts)
- Email addresses
- Dates and times
"""

from .extractors import (
    extract_all_entities,
    extract_persons,
    extract_organizations,
    extract_locations,
    extract_financial_entities,
    extract_email_addresses,
    extract_dates_times,
    extract_entities_from_dataframe,
    EntityExtractionResult
)
from .knowledge_base import (
    KnowledgeBase,
    PersonInfo,
    OrganizationInfo,
    create_knowledge_base_from_dataframe
)

# Optional imports (only if modules exist)
try:
    from .ner import extract_entities_with_ner
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    extract_entities_with_ner = None

try:
    from .linking import link_entities_across_emails
    LINKING_AVAILABLE = True
except ImportError:
    LINKING_AVAILABLE = False
    link_entities_across_emails = None

__all__ = [
    'extract_all_entities',
    'extract_persons',
    'extract_organizations',
    'extract_locations',
    'extract_financial_entities',
    'extract_email_addresses',
    'extract_dates_times',
    'extract_entities_from_dataframe',
    'EntityExtractionResult',
    'KnowledgeBase',
    'PersonInfo',
    'OrganizationInfo',
    'create_knowledge_base_from_dataframe',
]

if NER_AVAILABLE:
    __all__.append('extract_entities_with_ner')

if LINKING_AVAILABLE:
    __all__.append('link_entities_across_emails')

