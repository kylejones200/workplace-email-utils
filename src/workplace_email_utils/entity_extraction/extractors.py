"""
Regex-based entity extraction module.

Extracts entities using pattern matching before applying more sophisticated NER.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    """Container for extracted entities."""
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    financial_amounts: List[Dict] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    dates_times: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# Email address pattern (reused from email_parser)
EMAIL_PATTERN = r'[\w\.-]+@[\w\.-]+\.\w+'

# Financial amount patterns
MONEY_PATTERNS = [
    r'\$[\d,]+\.?\d*',  # $1,000 or $100.50
    r'[\d,]+\.?\d*\s*dollars?',  # 1,000 dollars
    r'[\d,]+\.?\d*\s*USD',  # 1,000 USD
    r'[\d,]+\.?\d*\s*million',  # 5 million
    r'[\d,]+\.?\d*\s*billion',  # 2 billion
]

# Date/time patterns (common formats in email)
DATE_PATTERNS = [
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
    r'\w+\s+\d{1,2},?\s+\d{4}',  # January 1, 2024
    r'\d{1,2}\s+\w+\s+\d{4}',  # 1 January 2024
    r'\w+\s+\d{1,2}',  # Monday 15
    r'\d{4}-\d{2}-\d{2}',  # 2024-01-15 (ISO format)
]

# Time patterns
TIME_PATTERNS = [
    r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)',  # 3:30 PM
    r'\d{1,2}:\d{2}',  # 15:30
    r'\d{1,2}\s*(?:AM|PM|am|pm)',  # 3 PM
]

# Common organization keywords
ORG_KEYWORDS = [
    r'\b(?:Inc\.?|Corp\.?|Corporation|LLC|Ltd\.?|Limited|Company|Co\.?)\b',
    r'\b(?:Department|Dept\.?|Division|Team|Group|Unit)\b',
    r'\b(?:Associates|Partners|Consulting|Solutions|Services)\b',
]

# Common location keywords
LOCATION_KEYWORDS = [
    r'\b(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?)\b',
    r'\b(?:City|Town|County|State|Province|Country)\b',
]


def extract_email_addresses(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to search for email addresses
        
    Returns:
        List of email addresses found
    """
    if not text:
        return []
    
    emails = re.findall(EMAIL_PATTERN, text, re.IGNORECASE)
    # Normalize (lowercase, remove duplicates)
    emails = list(set([e.lower().strip() for e in emails]))
    return emails


def extract_financial_entities(text: str) -> List[Dict]:
    """
    Extract financial amounts from text.
    
    Args:
        text: Text to search for financial amounts
        
    Returns:
        List of dictionaries with 'amount' and 'context' keys
    """
    if not text:
        return []
    
    amounts = []
    
    for pattern in MONEY_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amount_text = match.group(0)
            start, end = match.span()
            
            # Extract context (50 chars before and after)
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end].strip()
            
            # Try to extract numeric value
            numeric_value = None
            numeric_match = re.search(r'[\d,]+\.?\d*', amount_text)
            if numeric_match:
                numeric_str = numeric_match.group(0).replace(',', '')
                try:
                    numeric_value = float(numeric_str)
                    # Check for million/billion multipliers
                    if 'million' in amount_text.lower():
                        numeric_value *= 1_000_000
                    elif 'billion' in amount_text.lower():
                        numeric_value *= 1_000_000_000
                except ValueError:
                    pass
            
            amounts.append({
                'amount_text': amount_text,
                'numeric_value': numeric_value,
                'context': context,
                'position': (start, end)
            })
    
    # Remove duplicates (same amount text)
    seen = set()
    unique_amounts = []
    for amount in amounts:
        key = amount['amount_text']
        if key not in seen:
            seen.add(key)
            unique_amounts.append(amount)
    
    return unique_amounts


def extract_dates_times(text: str) -> List[str]:
    """
    Extract dates and times from text.
    
    Args:
        text: Text to search for dates and times
        
    Returns:
        List of date/time strings found
    """
    if not text:
        return []
    
    dates_times = []
    
    # Extract dates
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates_times.extend(matches)
    
    # Extract times
    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates_times.extend(matches)
    
    # Remove duplicates and normalize
    dates_times = list(set([dt.strip() for dt in dates_times]))
    return dates_times


def extract_persons(text: str, 
                    known_people: Optional[Set[str]] = None,
                    knowledge_base: Optional[object] = None) -> List[str]:
    """
    Extract person names from text using heuristics.
    
    This is a basic implementation. For better results, use NER (extract_entities_with_ner).
    
    Args:
        text: Text to search for person names
        known_people: Optional set of known person names to match (deprecated: use knowledge_base)
        knowledge_base: Optional KnowledgeBase object with known people/aliases
        
    Returns:
        List of person names found (canonical names when knowledge_base provided)
    """
    if not text:
        return []
    
    persons = []
    
    # Pattern: Title FirstName LastName (common in email signatures)
    title_pattern = r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
    matches = re.findall(title_pattern, text)
    persons.extend(matches)
    
    # Pattern: FirstName LastName (capitalized, 2-4 words)
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    matches = re.findall(name_pattern, text)
    
    # Filter out common false positives
    false_positives = {
        'Email', 'Subject', 'From', 'To', 'Date', 'Monday', 'Tuesday', 
        'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
        'Please', 'Thank', 'Best', 'Regards', 'Sincerely'
    }
    
    for match in matches:
        name = match.strip()
        # Filter if it's not a false positive and has reasonable length
        if name not in false_positives and 3 <= len(name) <= 40:
            # Check if it looks like a name (has at least 2 capitalized words)
            words = name.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words):
                persons.append(name)
    
    # Use knowledge base if provided (preferred over known_people)
    if knowledge_base:
        from .knowledge_base import KnowledgeBase
        known_people_set = knowledge_base.get_known_people()
        
        # Search for known people and their aliases
        for alias in known_people_set:
            if alias.lower() in text.lower():
                # Get canonical name
                canonical = knowledge_base.get_person_canonical(alias)
                if canonical and canonical not in persons:
                    persons.append(canonical)
    
    # Fallback to known_people if provided (for backwards compatibility)
    elif known_people:
        for person in known_people:
            if person.lower() in text.lower():
                persons.append(person)
    
    # Remove duplicates
    persons = list(set(persons))
    return persons


def extract_organizations(text: str, knowledge_base: Optional[object] = None) -> List[str]:
    """
    Extract organization names from text using patterns.
    
    Args:
        text: Text to search for organizations
        knowledge_base: Optional KnowledgeBase object with known organizations
        
    Returns:
        List of organization names found (canonical names when knowledge_base provided)
    """
    if not text:
        return []
    
    organizations = []
    
    # Pattern: Organization name followed by keywords
    for keyword_pattern in ORG_KEYWORDS:
        # Match text before organization keywords
        pattern = rf'([A-Z][A-Za-z\s&,.-]{{2,50}}?)\s*{keyword_pattern}'
        matches = re.findall(pattern, text)
        organizations.extend([m.strip() for m in matches])
    
    # Pattern: Common department patterns
    dept_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Department|Dept\.?|Division|Team)'
    matches = re.findall(dept_pattern, text)
    organizations.extend([m.strip() + ' Department' for m in matches])
    
    # Use knowledge base if provided
    if knowledge_base:
        from .knowledge_base import KnowledgeBase
        known_orgs_set = knowledge_base.get_known_organizations()
        
        # Search for known organizations and their aliases
        for alias in known_orgs_set:
            if alias.lower() in text.lower():
                # Get canonical name
                canonical = knowledge_base.get_org_canonical(alias)
                if canonical and canonical not in organizations:
                    organizations.append(canonical)
    
    # Remove duplicates and filter
    organizations = list(set([org.strip() for org in organizations if len(org.strip()) > 2]))
    return organizations


def extract_locations(text: str) -> List[str]:
    """
    Extract location names from text using patterns.
    
    Args:
        text: Text to search for locations
        
    Returns:
        List of location names found
    """
    if not text:
        return []
    
    locations = []
    
    # Pattern: Address patterns (Street, Avenue, etc.)
    for keyword_pattern in LOCATION_KEYWORDS:
        # Match text before location keywords
        pattern = rf'([A-Z0-9][A-Za-z0-9\s,.-]{{2,50}}?)\s*{keyword_pattern}'
        matches = re.findall(pattern, text)
        locations.extend([m.strip() for m in matches])
    
    # Pattern: City, State format
    city_state_pattern = r'([A-Z][a-z]+),?\s+([A-Z]{2})'
    matches = re.findall(city_state_pattern, text)
    locations.extend([f"{city}, {state}" for city, state in matches])
    
    # Pattern: Countries (common country names - simplified)
    countries = ['United States', 'USA', 'UK', 'United Kingdom', 'Canada', 
                 'Australia', 'Germany', 'France', 'Japan', 'China', 'India']
    for country in countries:
        if country in text:
            locations.append(country)
    
    # Remove duplicates
    locations = list(set([loc.strip() for loc in locations if len(loc.strip()) > 2]))
    return locations


def extract_all_entities(text: str, 
                         known_people: Optional[Set[str]] = None,
                         knowledge_base: Optional[object] = None) -> EntityExtractionResult:
    """
    Extract all entities from text.
    
    Args:
        text: Text to extract entities from
        known_people: Optional set of known person names (deprecated: use knowledge_base)
        knowledge_base: Optional KnowledgeBase object with known entities
        
    Returns:
        EntityExtractionResult with all extracted entities
    """
    result = EntityExtractionResult()
    
    result.email_addresses = extract_email_addresses(text)
    result.financial_amounts = extract_financial_entities(text)
    result.dates_times = extract_dates_times(text)
    result.persons = extract_persons(text, known_people, knowledge_base)
    result.organizations = extract_organizations(text, knowledge_base)
    result.locations = extract_locations(text)
    
    # Metadata
    result.metadata = {
        'total_entities': (
            len(result.email_addresses) +
            len(result.financial_amounts) +
            len(result.dates_times) +
            len(result.persons) +
            len(result.organizations) +
            len(result.locations)
        ),
        'text_length': len(text) if text else 0
    }
    
    return result


def extract_entities_from_dataframe(df: pd.DataFrame, 
                                    text_col: str = 'text',
                                    known_people: Optional[Set[str]] = None,
                                    knowledge_base: Optional[object] = None) -> pd.DataFrame:
    """
    Extract entities from all rows in a DataFrame.
    
    Args:
        df: DataFrame with email data
        text_col: Column name containing text to extract from
        known_people: Optional set of known person names (deprecated: use knowledge_base)
        knowledge_base: Optional KnowledgeBase object with known entities
        
    Returns:
        DataFrame with added entity columns
    """
    logger.info(f"Extracting entities from {len(df)} emails")
    
    if knowledge_base:
        logger.info(f"Using knowledge base: {len(knowledge_base.persons)} persons, {len(knowledge_base.organizations)} organizations")
    
    df_result = df.copy()
    
    # Initialize entity columns
    df_result['entities_persons'] = None
    df_result['entities_organizations'] = None
    df_result['entities_locations'] = None
    df_result['entities_financial'] = None
    df_result['entities_emails'] = None
    df_result['entities_dates'] = None
    
    results = []
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ''))
        if not text:
            results.append(EntityExtractionResult())
            continue
        
        result = extract_all_entities(text, known_people, knowledge_base)
        results.append(result)
        
        # Log progress
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} emails")
    
    # Add entity data to DataFrame
    df_result['entities_persons'] = [r.persons for r in results]
    df_result['entities_organizations'] = [r.organizations for r in results]
    df_result['entities_locations'] = [r.locations for r in results]
    df_result['entities_financial'] = [r.financial_amounts for r in results]
    df_result['entities_emails'] = [r.email_addresses for r in results]
    df_result['entities_dates'] = [r.dates_times for r in results]
    
    logger.info(f"Entity extraction complete")
    logger.info(f"  Total persons: {sum(len(r.persons) for r in results)}")
    logger.info(f"  Total organizations: {sum(len(r.organizations) for r in results)}")
    logger.info(f"  Total locations: {sum(len(r.locations) for r in results)}")
    logger.info(f"  Total emails: {sum(len(r.email_addresses) for r in results)}")
    
    return df_result


if __name__ == "__main__":
    # Test the extractors
    test_text = """
    Hi John Smith,
    
    Please review the contract for ABC Corporation. The amount is $1,500,000 USD.
    The meeting is scheduled for January 15, 2024 at 3:00 PM at 123 Main Street, Houston, TX.
    
    Contact: jane.doe@example.com or call our New York office.
    
    Best regards,
    Dr. Sarah Johnson
    """
    
    print("Testing entity extraction...")
    result = extract_all_entities(test_text)
    
    print(f"\nPersons: {result.persons}")
    print(f"Organizations: {result.organizations}")
    print(f"Locations: {result.locations}")
    print(f"Email addresses: {result.email_addresses}")
    print(f"Financial amounts: {len(result.financial_amounts)}")
    print(f"Dates/times: {result.dates_times}")
    print(f"Total entities: {result.metadata['total_entities']}")

