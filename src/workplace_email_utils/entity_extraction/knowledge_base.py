"""
Knowledge base for entity extraction.

Allows users to provide known entities, titles, aliases, etc. to improve extraction accuracy.
"""

import json
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersonInfo:
    """Information about a known person."""
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    title: Optional[str] = None
    email: Optional[str] = None
    organization: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class OrganizationInfo:
    """Information about a known organization."""
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    """Container for entity knowledge base."""
    persons: Dict[str, PersonInfo] = field(default_factory=dict)  # canonical_name -> PersonInfo
    organizations: Dict[str, OrganizationInfo] = field(default_factory=dict)  # canonical_name -> OrganizationInfo
    
    # Reverse lookups for fast access
    _person_aliases: Dict[str, str] = field(default_factory=dict)  # alias -> canonical_name
    _org_aliases: Dict[str, str] = field(default_factory=dict)  # alias -> canonical_name
    
    def __post_init__(self):
        """Build reverse lookup dictionaries."""
        self._rebuild_lookups()
    
    def _rebuild_lookups(self):
        """Rebuild reverse lookup dictionaries."""
        self._person_aliases = {}
        self._org_aliases = {}
        
        for canonical, person in self.persons.items():
            self._person_aliases[canonical.lower()] = canonical
            if person.aliases:
                for alias in person.aliases:
                    self._person_aliases[alias.lower()] = canonical
        
        for canonical, org in self.organizations.items():
            self._org_aliases[canonical.lower()] = canonical
            if org.aliases:
                for alias in org.aliases:
                    self._org_aliases[alias.lower()] = canonical
    
    def add_person(self, person: PersonInfo):
        """Add a person to the knowledge base."""
        canonical = person.canonical_name
        self.persons[canonical] = person
        self._person_aliases[canonical.lower()] = canonical
        if person.aliases:
            for alias in person.aliases:
                self._person_aliases[alias.lower()] = canonical
    
    def add_organization(self, org: OrganizationInfo):
        """Add an organization to the knowledge base."""
        canonical = org.canonical_name
        self.organizations[canonical] = org
        self._org_aliases[canonical.lower()] = canonical
        if org.aliases:
            for alias in org.aliases:
                self._org_aliases[alias.lower()] = canonical
    
    def get_person_canonical(self, name: str) -> Optional[str]:
        """Get canonical name for a person alias."""
        return self._person_aliases.get(name.lower())
    
    def get_org_canonical(self, name: str) -> Optional[str]:
        """Get canonical name for an organization alias."""
        return self._org_aliases.get(name.lower())
    
    def get_known_people(self) -> Set[str]:
        """Get set of all known person names and aliases."""
        names = set(self.persons.keys())
        for person in self.persons.values():
            names.update(person.aliases)
        return names
    
    def get_known_organizations(self) -> Set[str]:
        """Get set of all known organization names and aliases."""
        names = set(self.organizations.keys())
        for org in self.organizations.values():
            names.update(org.aliases)
        return names
    
    def to_dict(self) -> Dict:
        """Convert knowledge base to dictionary."""
        return {
            'persons': {k: asdict(v) for k, v in self.persons.items()},
            'organizations': {k: asdict(v) for k, v in self.organizations.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeBase':
        """Create knowledge base from dictionary."""
        kb = cls()
        
        if 'persons' in data:
            for canonical, info in data['persons'].items():
                person = PersonInfo(**info)
                kb.add_person(person)
        
        if 'organizations' in data:
            for canonical, info in data['organizations'].items():
                org = OrganizationInfo(**info)
                kb.add_organization(org)
        
        return kb
    
    def save(self, filepath: str):
        """Save knowledge base to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved knowledge base to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeBase':
        """Load knowledge base from JSON file."""
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"Knowledge base file not found: {filepath}. Returning empty knowledge base.")
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        kb = cls.from_dict(data)
        logger.info(f"Loaded knowledge base from {filepath}")
        logger.info(f"  Persons: {len(kb.persons)}")
        logger.info(f"  Organizations: {len(kb.organizations)}")
        
        return kb
    
    def merge(self, other: 'KnowledgeBase'):
        """Merge another knowledge base into this one."""
        for person in other.persons.values():
            self.add_person(person)
        
        for org in other.organizations.values():
            self.add_organization(org)


def create_knowledge_base_from_dataframe(df, 
                                         sender_col: str = 'sender',
                                         recipient_col: str = 'recipients') -> KnowledgeBase:
    """
    Create a knowledge base from email DataFrame by extracting known people.
    
    Args:
        df: DataFrame with email data
        sender_col: Column name for sender
        recipient_col: Column name for recipients
        
    Returns:
        KnowledgeBase with extracted people
    """
    kb = KnowledgeBase()
    
    # Extract unique senders
    senders = df[sender_col].dropna().unique()
    
    for sender_email in senders:
        if not sender_email or not isinstance(sender_email, str):
            continue
        
        # Extract name from email (before @)
        name_part = sender_email.split('@')[0]
        # Convert to readable name (e.g., john.doe -> John Doe)
        name = name_part.replace('.', ' ').replace('_', ' ').title()
        
        person = PersonInfo(
            canonical_name=name,
            email=sender_email.lower(),
            aliases=[name_part]  # Add email prefix as alias
        )
        kb.add_person(person)
    
    # Extract unique recipients
    recipients_set = set()
    for recipients in df[recipient_col].dropna():
        if isinstance(recipients, list):
            recipients_set.update(recipients)
        elif isinstance(recipients, str):
            recipients_set.add(recipients)
    
    for recipient_email in recipients_set:
        if not recipient_email or not isinstance(recipient_email, str):
            continue
        
        if recipient_email.lower() not in kb._person_aliases:
            name_part = recipient_email.split('@')[0]
            name = name_part.replace('.', ' ').replace('_', ' ').title()
            
            person = PersonInfo(
                canonical_name=name,
                email=recipient_email.lower(),
                aliases=[name_part]
            )
            kb.add_person(person)
    
    logger.info(f"Created knowledge base from DataFrame: {len(kb.persons)} persons")
    return kb


if __name__ == "__main__":
    # Example usage
    kb = KnowledgeBase()
    
    # Add a person with aliases and title
    kb.add_person(PersonInfo(
        canonical_name="John Smith",
        aliases=["John", "J. Smith", "Johnny"],
        title="Senior Vice President",
        email="john.smith@company.com",
        organization="ABC Corporation"
    ))
    
    # Add an organization
    kb.add_organization(OrganizationInfo(
        canonical_name="ABC Corporation",
        aliases=["ABC Corp", "ABC", "ABC Inc."],
        domain="abc.com"
    ))
    
    # Test lookups
    print(f"Canonical for 'Johnny': {kb.get_person_canonical('Johnny')}")
    print(f"Canonical for 'ABC': {kb.get_org_canonical('ABC')}")
    
    # Save and load
    kb.save("example_knowledge_base.json")
    kb_loaded = KnowledgeBase.load("example_knowledge_base.json")
    print(f"Loaded: {len(kb_loaded.persons)} persons")

