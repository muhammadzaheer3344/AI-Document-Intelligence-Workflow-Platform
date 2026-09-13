"""
generate_dataset.py
--------------------
Week 3 asks us to "prepare a cleaner, reasonably balanced dataset" of the
document classes the project supports (Invoice, Resume, Other).

IMPORTANT (read this before you submit): this script generates *synthetic*
text examples from templates so the pipeline has something real to train
and evaluate on out of the box. For your actual submission, swap in real
sample invoices/resumes you collect (see README "Using your own data") —
graders want to see it work on real-world documents, not just templates.
Keep this generator around though, it's a legitimate way to pad/balance
classes if you're short on real examples of one type.

Run: python data/generate_dataset.py
Produces: data/dataset.csv  (columns: text, label)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

COMPANIES = ["ABC Technologies", "Sunrise Traders", "Nexus Logistics", "Pioneer Textiles",
             "BlueSky Electronics", "Metro Hardware", "Falcon Freight", "Greenfield Foods",
             "Orbit Solutions", "Alpine Consultants", "Zenith Retailers", "Summit Motors"]

ITEMS = ["laptop repair service", "office stationery supply", "consulting hours",
         "spare parts (screen assembly)", "bulk cotton fabric", "network cabling",
         "monthly cloud hosting", "furniture delivery", "printer cartridges",
         "software license renewal", "logistics handling fee", "raw material batch"]

NAMES = ["Ayesha Khan", "Bilal Ahmed", "Sara Malik", "Hamza Raza", "Fatima Noor",
         "Usman Tariq", "Zainab Siddiqui", "Ali Hassan", "Mahnoor Iqbal", "Omar Farooq",
         "Hira Shah", "Talha Javed"]

SKILLS_POOL = ["Python", "SQL", "Machine Learning", "Data Analysis", "Excel", "Java",
               "Project Management", "Communication", "React", "Node.js", "AWS",
               "Deep Learning", "Customer Service", "Sales", "Accounting", "Marketing",
               "TensorFlow", "Docker", "Linux", "Power BI"]

OTHER_SNIPPETS = [
    """Meeting Minutes - Weekly Sync
Date: {date}
Attendees discussed project timelines and budget allocations for the upcoming quarter.
Action items were assigned to team leads. Next meeting scheduled in two weeks.""",
    """Terms and Conditions
This agreement is entered into by the parties listed below. All disputes shall be
resolved through arbitration. This document does not constitute a purchase order
or a statement of employment history.""",
    """University Lecture Notes - Introduction to Databases
Topic: Normalization forms (1NF, 2NF, 3NF)
A relation is in 1NF if all attributes contain atomic values. Today's lecture also
covered functional dependencies and candidate keys.""",
    """Weather Advisory
Heavy rainfall is expected across the region over the next 48 hours. Residents are
advised to avoid low-lying areas. Local authorities have issued a flood warning.""",
    """Recipe: Chicken Biryani
Ingredients include basmati rice, chicken, yogurt, and a blend of spices. Marinate
the chicken for at least two hours before cooking on low heat.""",
]


def random_date() -> str:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.choice([2024, 2025, 2026])
    return f"{day:02d}-{month:02d}-{year}"


def make_invoice() -> str:
    inv_no = f"INV-{random.randint(1000, 9999)}"
    company = random.choice(COMPANIES)
    date = random_date()
    n_items = random.randint(1, 3)
    lines = []
    subtotal = 0
    for _ in range(n_items):
        item = random.choice(ITEMS)
        qty = random.randint(1, 10)
        price = random.randint(500, 20000)
        line_total = qty * price
        subtotal += line_total
        lines.append(f"{item} x{qty} @ Rs.{price} = Rs.{line_total}")
    tax = round(subtotal * 0.17)
    total = subtotal + tax
    return f"""INVOICE
Invoice Number: {inv_no}
Date: {date}
Bill To: {company}
Company Name: {company}

Description of Goods/Services:
{chr(10).join(lines)}

Subtotal: Rs.{subtotal}
Tax (17%): Rs.{tax}
Total Amount: Rs.{total}

Payment due within 15 days of invoice date. Thank you for your business."""


def make_resume() -> str:
    name = random.choice(NAMES)
    email = name.lower().replace(" ", ".") + random.choice(["@gmail.com", "@outlook.com", "@yahoo.com"])
    phone = f"03{random.randint(0,9)}{random.randint(1000000,9999999)}"
    skills = random.sample(SKILLS_POOL, k=random.randint(4, 7))
    years = random.randint(1, 8)
    return f"""RESUME

{name}
Email: {email}
Phone: {phone}

OBJECTIVE
Motivated professional with {years} years of experience seeking a challenging role
to apply technical and analytical skills.

SKILLS
{', '.join(skills)}

EXPERIENCE
Software Developer, Previous Company ({2026 - years}-2026)
- Worked on backend systems and data pipelines.
- Collaborated with cross-functional teams to deliver projects on schedule.

EDUCATION
Bachelor of Science in Computer Science, University (Graduated {2026 - years - 4})

REFERENCES
Available upon request."""


def make_other() -> str:
    template = random.choice(OTHER_SNIPPETS)
    return template.format(date=random_date())


def build_dataset(n_per_class: int = 60) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    generators = {"Invoice": make_invoice, "Resume": make_resume, "Other": make_other}
    for label, gen in generators.items():
        for _ in range(n_per_class):
            rows.append((gen(), label))
    random.shuffle(rows)
    return rows


def main() -> None:
    rows = build_dataset(n_per_class=60)
    out_path = Path(__file__).parent / "dataset.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print("Class balance:", counts)


if __name__ == "__main__":
    main()
