"""
Named Entity Recognition (NER) using spaCy or other NLP libraries.

Provides more accurate entity extraction than regex patterns.
"""

import re
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NERResult:
    """Container for NER results."""
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    other: List[Dict]  # Other entity types with labels


# Global model cache
_spacy_model = None
_transformers_ner = None


def load_spacy_model(model_name: str = 'en_core_web_sm') -> Optional[object]:
    """
    Load spaCy NER model.
    
    Args:
        model_name: Name of spaCy model to load
        
    Returns:
        Loaded spaCy model or None
    """
    global _spacy_model
    
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available. Using regex fallback.")
        return None
    
    if _spacy_model is None:
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            _spacy_model = spacy.load(model_name)
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            return None
    
    return _spacy_model


def load_transformers_ner(model_name: str = 'dslim/bert-base-NER') -> Optional[object]:
    """
    Load transformers NER pipeline.
    
    Args:
        model_name: Name of transformers model
        
    Returns:
        NER pipeline or None
    """
    global _transformers_ner
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not available. Using regex fallback.")
        return None
    
    if _transformers_ner is None:
        try:
            logger.info(f"Loading transformers NER model: {model_name}")
            _transformers_ner = pipeline('ner', model=model_name, aggregation_strategy='simple')
            logger.info("Transformers NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load transformers NER: {e}")
            return None
    
    return _transformers_ner


def extract_entities_with_spacy(text: str, model_name: str = 'en_core_web_sm') -> NERResult:
    """
    Extract entities using spaCy NER.
    
    Args:
        text: Text to extract entities from
        model_name: spaCy model name
        
    Returns:
        NERResult with extracted entities
    """
    if not text:
        return NERResult(persons=[], organizations=[], locations=[], other=[])
    
    nlp = load_spacy_model(model_name)
    if nlp is None:
        # Fallback to regex
        from .extractors import extract_all_entities
        result = extract_all_entities(text)
        return NERResult(
            persons=result.persons,
            organizations=result.organizations,
            locations=result.locations,
            other=[]
        )
    
    doc = nlp(text)
    
    persons = []
    organizations = []
    locations = []
    other = []
    
    for ent in doc.ents:
        entity_text = ent.text.strip()
        
        if ent.label_ in ['PERSON']:
            persons.append(entity_text)
        elif ent.label_ in ['ORG', 'ORGANIZATION']:
            organizations.append(entity_text)
        elif ent.label_ in ['GPE', 'LOC', 'LOCATION']:
            locations.append(entity_text)
        else:
            other.append({
                'text': entity_text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    # Remove duplicates
    persons = list(set(persons))
    organizations = list(set(organizations))
    locations = list(set(locations))
    
    return NERResult(
        persons=persons,
        organizations=organizations,
        locations=locations,
        other=other
    )


def extract_entities_with_transformers(text: str, model_name: str = 'dslim/bert-base-NER') -> NERResult:
    """
    Extract entities using transformers NER pipeline.
    
    Args:
        text: Text to extract entities from
        model_name: Transformers model name
        
    Returns:
        NERResult with extracted entities
    """
    if not text:
        return NERResult(persons=[], organizations=[], locations=[], other=[])
    
    ner_pipeline = load_transformers_ner(model_name)
    if ner_pipeline is None:
        # Fallback to regex
        from .extractors import extract_all_entities
        result = extract_all_entities(text)
        return NERResult(
            persons=result.persons,
            organizations=result.organizations,
            locations=result.locations,
            other=[]
        )
    
    try:
        entities = ner_pipeline(text)
        
        persons = []
        organizations = []
        locations = []
        other = []
        
        for entity in entities:
            entity_text = entity['word'].strip()
            label = entity['entity_group']
            
            if label == 'PER':
                persons.append(entity_text)
            elif label == 'ORG':
                organizations.append(entity_text)
            elif label == 'LOC':
                locations.append(entity_text)
            else:
                other.append({
                    'text': entity_text,
                    'label': label,
                    'score': entity.get('score', 0.0)
                })
        
        # Remove duplicates
        persons = list(set(persons))
        organizations = list(set(organizations))
        locations = list(set(locations))
        
        return NERResult(
            persons=persons,
            organizations=organizations,
            locations=locations,
            other=other
        )
    
    except Exception as e:
        logger.warning(f"Transformers NER failed: {e}. Using regex fallback.")
        from .extractors import extract_all_entities
        result = extract_all_entities(text)
        return NERResult(
            persons=result.persons,
            organizations=result.organizations,
            locations=result.locations,
            other=[]
        )


def extract_entities_with_ner(text: str, method: str = 'spacy', **kwargs) -> NERResult:
    """
    Extract entities using NER (spaCy or transformers).
    
    Args:
        text: Text to extract entities from
        method: 'spacy' or 'transformers'
        **kwargs: Additional arguments for specific methods
        
    Returns:
        NERResult with extracted entities
    """
    if method == 'spacy':
        model_name = kwargs.get('model_name', 'en_core_web_sm')
        return extract_entities_with_spacy(text, model_name)
    elif method == 'transformers':
        model_name = kwargs.get('model_name', 'dslim/bert-base-NER')
        return extract_entities_with_transformers(text, model_name)
    else:
        logger.warning(f"Unknown NER method: {method}. Using regex fallback.")
        from .extractors import extract_all_entities
        result = extract_all_entities(text)
        return NERResult(
            persons=result.persons,
            organizations=result.organizations,
            locations=result.locations,
            other=[]
        )


if __name__ == "__main__":
    # Test NER
    test_text = """
    John Smith from ABC Corporation will be visiting our Houston office on January 15, 2024.
    Contact him at john.smith@abc.com for the $1.5 million contract details.
    """
    
    print("Testing NER extraction...")
    
    # Try spaCy first
    if SPACY_AVAILABLE:
        print("\nUsing spaCy:")
        result = extract_entities_with_spacy(test_text)
        print(f"Persons: {result.persons}")
        print(f"Organizations: {result.organizations}")
        print(f"Locations: {result.locations}")
    
    # Try transformers
    if TRANSFORMERS_AVAILABLE:
        print("\nUsing Transformers:")
        result = extract_entities_with_transformers(test_text)
        print(f"Persons: {result.persons}")
        print(f"Organizations: {result.organizations}")
        print(f"Locations: {result.locations}")
    
    # Fallback to regex
    print("\nUsing Regex (fallback):")
    from .extractors import extract_all_entities
    result = extract_all_entities(test_text)
    print(f"Persons: {result.persons}")
    print(f"Organizations: {result.organizations}")
    print(f"Locations: {result.locations}")

