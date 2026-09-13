import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.extract_text import extract_text
from src.preprocess import clean_text, normalize_for_classification, is_usable
from src.classifier import classify_document
from src.field_extraction import extract_fields, get_missing_fields

data = Path('sample_docs/scanned_invoice.png').read_bytes()
ext = extract_text(data, 'png')
print('Extraction method:', ext.method, '| success:', ext.success)
print('Warnings:', ext.warnings if ext.warnings else 'none')
cleaned = clean_text(ext.text)
print('Usable text:', is_usable(cleaned))
print('Text preview:', cleaned[:200])
if is_usable(cleaned):
    normalized = normalize_for_classification(cleaned)
    cls = classify_document(cleaned, normalized)
    print('Classification:', cls.label, '| confidence:', cls.confidence)
    fields = extract_fields(cleaned, cls.label)
    missing = get_missing_fields(fields)
    print('Fields:')
    for k, v in fields.items():
        print(f'  {k}: {v}')
    print('Missing fields:', missing if missing else 'none')
