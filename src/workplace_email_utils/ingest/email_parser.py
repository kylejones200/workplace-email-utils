"""
Email parsing module for unstructured Enron dataset.

Handles parsing of raw email messages from CSV format and maildir format into structured format.
"""

import re
import pandas as pd
from typing import List, Dict, Optional
from email.parser import Parser
from email.policy import default
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_email_message(message: str) -> Dict[str, any]:
    """
    Parse a raw email message string into structured components.
    
    Args:
        message: Raw email message string with headers
        
    Returns:
        Dictionary with parsed email fields
    """
    try:
        # Use email parser to handle headers properly
        msg = Parser(policy=default).parsestr(message)
        
        # Extract headers
        sender = msg.get('From', '').strip()
        to = msg.get('To', '').strip()
        cc = msg.get('Cc', '').strip()
        bcc = msg.get('Bcc', '').strip()
        subject = msg.get('Subject', '').strip()
        date = msg.get('Date', '').strip()
        message_id = msg.get('Message-ID', '').strip()
        
        # Threading headers (for Phase 1.3)
        in_reply_to = msg.get('In-Reply-To', '').strip()
        references = msg.get('References', '').strip()
        reply_to = msg.get('Reply-To', '').strip()
        
        # X-Headers (often more structured)
        x_from = msg.get('X-From', '').strip() or sender
        x_to = msg.get('X-To', '').strip() or to
        x_cc = msg.get('X-cc', '').strip() or cc
        x_bcc = msg.get('X-bcc', '').strip() or bcc
        x_folder = msg.get('X-Folder', '').strip()
        x_origin = msg.get('X-Origin', '').strip()
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(msg.get_payload())
        
        # Parse recipients
        recipients = []
        for field in [to, cc, bcc]:
            if field:
                # Extract email addresses using regex
                emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', field)
                recipients.extend(emails)
        
        # Normalize email addresses
        sender = normalize_email(sender)
        recipients = [normalize_email(r) for r in recipients if r]
        
        # Parse References header (space-separated Message-IDs)
        references_list = []
        if references:
            references_list = [ref.strip() for ref in references.split() if ref.strip()]
        
        return {
            'sender': sender,
            'recipients': list(set(recipients)),  # Remove duplicates
            'subject': subject,
            'body': body.strip(),
            'date': date,
            'message_id': message_id,
            'text': f"{subject}\n{body}".strip(),  # Combined text for analysis
            # Threading headers (Phase 1.3)
            'in_reply_to': in_reply_to,
            'references': references,
            'references_list': references_list,  # Parsed list of Message-IDs
            'reply_to': reply_to,
            # X-Headers
            'x_from': x_from,
            'x_to': x_to,
            'x_cc': x_cc,
            'x_bcc': x_bcc,
            'x_folder': x_folder,
            'x_origin': x_origin,
        }
    except Exception as e:
        logger.warning(f"Failed to parse email: {e}")
        # Fallback: try to extract basic info with regex
        return parse_email_fallback(message)


def normalize_email(email_str: str) -> str:
    """
    Normalize email address from various formats.
    
    Handles formats like:
    - "John Doe <john@example.com>"
    - "john@example.com"
    - "John Doe/Enron@EnronXGate"
    """
    if not email_str:
        return ""
    
    # Extract email from angle brackets
    match = re.search(r'<([^>]+)>', email_str)
    if match:
        return match.group(1).lower().strip()
    
    # Extract email directly
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_str)
    if match:
        return match.group(0).lower().strip()
    
    # Return cleaned version
    return email_str.lower().strip()


def parse_email_fallback(message: str) -> Dict[str, any]:
    """
    Fallback parser using regex when email parser fails.
    """
    sender = ""
    recipients = []
    subject = ""
    body = message
    
    # Try to extract headers with regex
    sender_match = re.search(r'From:\s*(.+?)(?:\n|$)', message, re.IGNORECASE)
    if sender_match:
        sender = normalize_email(sender_match.group(1))
    
    to_match = re.search(r'To:\s*(.+?)(?:\n|$)', message, re.IGNORECASE)
    if to_match:
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', to_match.group(1))
        recipients.extend([normalize_email(e) for e in emails])
    
    subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', message, re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()
    
    # Body is everything after first blank line or after headers
    body_match = re.search(r'\n\n(.+)', message, re.DOTALL)
    if body_match:
        body = body_match.group(1).strip()
    
    return {
        'sender': sender,
        'recipients': list(set(recipients)),
        'subject': subject,
        'body': body,
        'date': '',
        'message_id': '',
        'text': f"{subject}\n{body}".strip()
    }


def load_enron_csv(csv_path: str, max_rows: Optional[int] = None, 
                   sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and parse Enron emails from CSV file.
    
    Args:
        csv_path: Path to emails.csv file
        max_rows: Maximum number of rows to process (None for all)
        sample_size: Random sample size (None for no sampling)
        
    Returns:
        DataFrame with parsed email data
    """
    logger.info(f"Loading emails from {csv_path}")
    
    # Read CSV in chunks if it's very large
    chunk_size = 10000
    chunks = []
    rows_processed = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=max_rows):
        logger.info(f"Processing chunk: {rows_processed} rows")
        
        parsed_rows = []
        for idx, row in chunk.iterrows():
            try:
                file_path = row.get('file', '')
                message = row.get('message', '')
                
                if pd.isna(message) or not message:
                    continue
                
                parsed = parse_email_message(str(message))
                parsed['doc_id'] = file_path
                parsed_rows.append(parsed)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        if parsed_rows:
            chunk_df = pd.DataFrame(parsed_rows)
            chunks.append(chunk_df)
        
        rows_processed += len(chunk)
        if max_rows and rows_processed >= max_rows:
            break
    
    if not chunks:
        raise ValueError("No emails were successfully parsed")
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Apply sampling if requested
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} rows from {len(df)} total")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} emails")
    return df


def extract_folder_metadata(file_path: str, maildir_root: str) -> Dict[str, str]:
    """
    Extract folder metadata from maildir file path.
    
    Args:
        file_path: Full path to email file
        maildir_root: Root directory of maildir structure
        
    Returns:
        Dictionary with folder metadata:
        - user_folder: User's folder name (e.g., "allen-p")
        - folder_type: Type of folder (e.g., "inbox", "sent", "deleted_items")
        - folder_path: Relative path within user's maildir
    """
    # Get relative path from maildir root
    try:
        rel_path = os.path.relpath(file_path, maildir_root)
        path_parts = Path(rel_path).parts
        
        if len(path_parts) < 2:
            return {
                'user_folder': '',
                'folder_type': 'unknown',
                'folder_path': rel_path
            }
        
        user_folder = path_parts[0]
        
        # Common folder types to recognize
        folder_name = path_parts[1].lower()
        
        # Map common folder names to standardized types
        folder_type_mapping = {
            'inbox': 'inbox',
            'sent': 'sent',
            'sent_items': 'sent',
            '_sent_mail': 'sent',
            'deleted_items': 'deleted',
            'trash': 'deleted',
            'drafts': 'drafts',
            'all_documents': 'documents',
            'notes_inbox': 'notes',
            'discussion_threads': 'discussions',
        }
        
        # Check if folder name matches known types
        folder_type = folder_type_mapping.get(folder_name, folder_name)
        
        # Get full folder path (everything after user folder)
        folder_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else folder_name
        
        return {
            'user_folder': user_folder,
            'folder_type': folder_type,
            'folder_path': folder_path
        }
    except Exception as e:
        logger.warning(f"Failed to extract folder metadata from {file_path}: {e}")
        return {
            'user_folder': '',
            'folder_type': 'unknown',
            'folder_path': ''
        }


def load_enron_maildir(maildir_path: str, 
                       max_emails: Optional[int] = None,
                       sample_size: Optional[int] = None,
                       user_filter: Optional[List[str]] = None,
                       folder_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and parse Enron emails from maildir directory structure.
    
    Args:
        maildir_path: Path to maildir root directory
        max_emails: Maximum number of emails to process (None for all)
        sample_size: Random sample size (None for no sampling)
        user_filter: Optional list of user folders to include (e.g., ['allen-p', 'arnold-j'])
        folder_filter: Optional list of folder types to include (e.g., ['inbox', 'sent'])
        
    Returns:
        DataFrame with parsed email data including folder metadata
    """
    logger.info(f"Loading emails from maildir: {maildir_path}")
    
    maildir_root = Path(maildir_path).resolve()
    if not maildir_root.exists():
        raise ValueError(f"Maildir path does not exist: {maildir_path}")
    
    parsed_rows = []
    emails_processed = 0
    
    # Walk through maildir structure
    for user_folder in sorted(maildir_root.iterdir()):
        if not user_folder.is_dir():
            continue
        
        user_name = user_folder.name
        
        # Apply user filter if provided
        if user_filter and user_name not in user_filter:
            continue
        
        logger.info(f"Processing user: {user_name}")
        
        # Walk through subdirectories (folders like inbox, sent, etc.)
        for folder_path in user_folder.rglob('*'):
            if not folder_path.is_file():
                continue
            
            # Extract folder metadata
            folder_meta = extract_folder_metadata(str(folder_path), str(maildir_root))
            
            # Apply folder filter if provided
            if folder_filter and folder_meta['folder_type'] not in folder_filter:
                continue
            
            # Read and parse email file
            try:
                with open(folder_path, 'r', encoding='utf-8', errors='ignore') as f:
                    message_content = f.read()
                
                if not message_content.strip():
                    continue
                
                # Parse email message
                parsed = parse_email_message(message_content)
                
                # Add folder metadata and file path
                parsed['doc_id'] = str(folder_path.relative_to(maildir_root))
                parsed['user_folder'] = folder_meta['user_folder']
                parsed['folder_type'] = folder_meta['folder_type']
                parsed['folder_path'] = folder_meta['folder_path']
                parsed['file_path'] = str(folder_path)
                
                parsed_rows.append(parsed)
                emails_processed += 1
                
                # Log progress every 1000 emails
                if emails_processed % 1000 == 0:
                    logger.info(f"Processed {emails_processed} emails...")
                
                # Check max_emails limit
                if max_emails and emails_processed >= max_emails:
                    logger.info(f"Reached max_emails limit: {max_emails}")
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing file {folder_path}: {e}")
                continue
        
        # Break if we've reached max_emails
        if max_emails and emails_processed >= max_emails:
            break
    
    if not parsed_rows:
        raise ValueError("No emails were successfully parsed from maildir")
    
    # Create DataFrame
    df = pd.DataFrame(parsed_rows)
    
    # Apply sampling if requested
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} rows from {len(df)} total")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} emails from maildir")
    logger.info(f"Folder type distribution:\n{df['folder_type'].value_counts()}")
    
    return df


def load_emails(data_path: str,
                data_format: str = 'auto',
                max_rows: Optional[int] = None,
                sample_size: Optional[int] = None,
                user_filter: Optional[List[str]] = None,
                folder_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Unified function to load emails from either CSV or maildir format.
    
    Args:
        data_path: Path to emails.csv file or maildir directory
        data_format: 'csv', 'maildir', or 'auto' (detects based on path)
        max_rows: Maximum number of rows/emails to process (None for all)
        sample_size: Random sample size (None for no sampling)
        user_filter: Optional list of user folders to include (maildir only)
        folder_filter: Optional list of folder types to include (maildir only)
        
    Returns:
        DataFrame with parsed email data
    """
    # Auto-detect format if not specified
    if data_format == 'auto':
        if os.path.isdir(data_path):
            data_format = 'maildir'
        elif data_path.endswith('.csv'):
            data_format = 'csv'
        else:
            # Try to detect
            if os.path.exists(data_path) and os.path.isfile(data_path):
                data_format = 'csv'
            elif os.path.exists(data_path) and os.path.isdir(data_path):
                data_format = 'maildir'
            else:
                raise ValueError(f"Could not auto-detect format for: {data_path}")
    
    logger.info(f"Loading emails from {data_format} format: {data_path}")
    
    if data_format == 'csv':
        return load_enron_csv(data_path, max_rows=max_rows, sample_size=sample_size)
    elif data_format == 'maildir':
        return load_enron_maildir(
            data_path, 
            max_emails=max_rows,
            sample_size=sample_size,
            user_filter=user_filter,
            folder_filter=folder_filter
        )
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Use 'csv' or 'maildir'")


if __name__ == "__main__":
    # Test the parser
    test_message = """Message-ID: <18782981.1075855378110.JavaMail.evans@thyme>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: phillip.allen@enron.com
To: tim.belden@enron.com
Subject: Forecast
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii

Here is our forecast
"""
    
    result = parse_email_message(test_message)
    print(result)

