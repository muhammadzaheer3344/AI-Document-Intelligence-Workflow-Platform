import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.extract_text import extract_text
from src.preprocess import clean_text, normalize_for_classification
from src.classifier import classify_document
from src.field_extraction import extract_fields, get_missing_fields

data = Path('sample_docs/native_invoice.pdf').read_bytes()
ext = extract_text(data, 'pdf')
print('Extraction method:', ext.method, '| success:', ext.success)
cleaned = clean_text(ext.text)
normalized = normalize_for_classification(cleaned)
cls = classify_document(cleaned, normalized)
print('Classification:', cls.label, '| confidence:', cls.confidence, '| method:', cls.method)
fields = extract_fields(cleaned, cls.label)
missing = get_missing_fields(fields)
print('Fields:')
for k, v in fields.items():
    print(f'  {k}: {v}')
print('Missing fields:', missing if missing else 'none')
