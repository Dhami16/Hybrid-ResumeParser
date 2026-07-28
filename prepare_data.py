"""
prepare_data.py — Phase 3: expanded synthetic training data generator

Replaces the Week 8 stub, which only defined generate_contextual_data() and
never actually wrote train.spacy/dev.spacy (train.spacy/dev.spacy in the repo
were produced by an earlier, undocumented version of this script). This
rewrite addresses the two problems Phase 2 exposed:

  1. dev.spacy was sampled from the same ~500-combination template pool as
     train.spacy (5 names x 4 universities x 5 templates), so the "1.0 F1"
     eval score measured memorization, not generalization. Fixed here by
     holding out a slice of names, universities, and negative-example pools
     entirely for dev — values dev never sees during train.
  2. The trained model fired NAME/UNIVERSITY on section headers, job titles,
     company names, and skill/location lines from real resumes, because
     training data never showed it what those look like. Fixed by adding
     realistic multi-line "mini resume" documents that include those exact
     distractor types as explicit zero-entity negative examples.

Usage:
    python prepare_data.py [--train-count N] [--dev-count N] [--seed N]
"""

import argparse
import random

import spacy
from spacy.tokens import DocBin

# ── Entity value pools ─────────────────────────────────────────────────────

NAMES = [
    "Rahul Verma", "Amit Sharma", "Priya Singh", "Anjali Gupta", "Vikram Malhotra",
    "Meera Chandrasekaran", "Kevin Liang", "Bella Trevino", "Arjun Nair", "Sneha Reddy",
    "Karthik Iyer", "Divya Menon", "Rohan Kapoor", "Ishaan Bose", "Ananya Rao",
    "Wei Zhang", "Li Wei", "Sofia Martinez", "Lucas Silva", "Emma Johnson",
    "Noah Williams", "Olivia Brown", "Aditya Kumar", "Neha Joshi", "Siddharth Pillai",
    "Tanvi Deshmukh", "Kabir Khanna", "Riya Chawla", "Yusuf Ansari", "Fatima Sheikh",
    "Chen Jing", "Hiroshi Tanaka", "Yuki Sato", "Daniel Kim", "Grace Park",
    "Michael Brooks", "Sarah Thompson", "James Anderson", "Aarav Mehta", "Kavya Iyer",
]

UNIVERSITIES = [
    "Thapar Institute", "IIT Delhi", "IIT Bombay", "BITS Pilani", "Delhi University",
    "Vellore Institute of Technology", "University of California, Davis",
    "University of Illinois Chicago", "NIT Trichy", "NIT Warangal",
    "Anna University", "Manipal Institute of Technology", "Stanford University",
    "Massachusetts Institute of Technology", "University of Michigan",
    "Georgia Institute of Technology", "Carnegie Mellon University",
    "SRM Institute of Science and Technology", "VIT Chennai", "Amity University",
]

# ── Negative-example pools: things that must NEVER be tagged as an entity ──
# Each pool targets a failure mode actually observed in Phase 2's tier trace.

SECTION_HEADERS = [
    "EDUCATION", "WORK EXPERIENCE", "TECHNICAL SKILLS", "PROJECTS",
    "CERTIFICATIONS", "CONTACT", "PROFILE", "CAREER OBJECTIVE", "SKILLS",
    "ACHIEVEMENTS", "PUBLICATIONS", "SUMMARY", "EXPERIENCE", "OBJECTIVE",
]

JOB_TITLES = [
    "Software Engineer", "Data Engineering Intern", "Web Developer",
    "Backend Developer", "Machine Learning Engineer", "Product Manager",
    "DevOps Engineer", "Full Stack Developer", "Research Assistant",
    "Teaching Assistant", "Business Analyst", "QA Engineer",
]

COMPANY_NAMES = [
    "Nimbus Analytics Pvt Ltd", "Solstice Labs", "Infosys Limited",
    "Tata Consultancy Services", "Wipro Technologies", "Amazon Web Services",
    "Microsoft Azure", "Zoho Corporation", "Freshworks Inc", "Google Cloud",
]

CERTIFICATIONS = [
    "AWS Certified Solutions Architect - Associate",
    "Google Cloud Professional Data Engineer",
    "Certified Kubernetes Administrator",
    "Microsoft Certified Azure Developer Associate",
    "Oracle Certified Java Programmer",
]

SKILL_LINES = [
    "Languages: Python, Java, SQL, TypeScript, Bash",
    "AI/ML Frameworks: TensorFlow, PyTorch, scikit-learn, Keras",
    "Cloud & Infrastructure: AWS, GCP, Azure, Firebase",
    "Databases: MySQL, PostgreSQL, MongoDB, Redis",
    "DevOps & Tools: Git, Docker, Kubernetes, Jenkins, CI/CD",
    "Soft Skills: Leadership, Communication, Teamwork, Agile",
]

PROJECT_BULLETS = [
    "Built ETL pipelines processing 2M+ records daily using Python and Airflow",
    "Migrated legacy monolith to microservices on AWS, reducing latency by 35%",
    "Developed a full-stack app used by 500+ students",
    "Deployed containerized services via Docker on GCP Cloud Run",
    "Automated report generation with Python, saving 10 hours per week",
    "Mentored 2 junior engineers and led sprint planning for a 6-person team",
]

LOCATION_FRAGMENTS = [
    "Bengaluru, Karnataka", "Davis, CA", "Chicago, IL", "Mumbai, Maharashtra",
    "New York, NY", "San Francisco, CA", "Pune, Maharashtra", "Austin, TX",
]

CONTACT_LINES = [
    "test.user@gmail.com | +91-98765-43210",
    "github.com/testuser | linkedin.com/in/testuser",
    "Phone: (123) 456-7890",
    "email: candidate@example.com",
]

CODE_SNIPPETS = [
    "import spacy\nnlp = spacy.load('en_core_web_sm')",
    "SELECT * FROM users WHERE active = 1;",
    "def parse(text):\n    return text.strip()",
]

NEGATIVE_POOLS = {
    "section_headers":    SECTION_HEADERS,
    "job_titles":         JOB_TITLES,
    "company_names":      COMPANY_NAMES,
    "certifications":     CERTIFICATIONS,
    "skill_lines":         SKILL_LINES,
    "project_bullets":    PROJECT_BULLETS,
    "location_fragments": LOCATION_FRAGMENTS,
    "contact_lines":      CONTACT_LINES,
    "code_snippets":      CODE_SNIPPETS,
}

# ── Sentence templates (short-form positive examples) ──────────────────────

NAME_INTROS = ["Resume of ", "I am ", "Name: ", "Curriculum Vitae: ", "Contact: ", ""]
UNI_INTROS = [
    "Studied at ", "Education from ", "Degree at ", "B.Tech from ", "Student of ",
    "Graduated from ", "Currently pursuing a degree at ", "",
]
DEGREE_LINES = [
    "B.Tech in Computer Science", "B.E. in Information Technology",
    "B.Sc in Data Science", "M.Tech in Computer Science",
    "Bachelor of Technology", "Master of Science in Computer Engineering",
]


def _split_pool(pool, train_ratio, rng):
    """Shuffle and split a pool so dev gets values never seen in train."""
    shuffled = pool[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * train_ratio))
    train_part = shuffled[:cut]
    dev_part = shuffled[cut:] or shuffled[-1:]
    return train_part, dev_part


def _build_text(lines_with_labels):
    """[(line_text, label_or_None), ...] -> (full_text, [(start, end, label), ...])"""
    parts, entities, cursor = [], [], 0
    for line, label in lines_with_labels:
        start = cursor
        end = start + len(line)
        if label:
            entities.append((start, end, label))
        parts.append(line)
        cursor = end + 1  # account for the "\n" joining lines
    return "\n".join(parts), entities


def generate_sentence_example(names, unis, rng):
    """Short-form: a sentence or two combining NAME + UNIVERSITY."""
    name = rng.choice(names)
    uni = rng.choice(unis)
    intro = rng.choice(NAME_INTROS)
    uni_intro = rng.choice(UNI_INTROS)
    text = f"{intro}{name}. {uni_intro}{uni}."
    name_start = len(intro)
    name_end = name_start + len(name)
    uni_start = text.find(uni)
    uni_end = uni_start + len(uni)
    return text, [(name_start, name_end, "NAME"), (uni_start, uni_end, "UNIVERSITY")]


def generate_mini_resume_example(names, unis, negatives, rng):
    """
    Longer-form: a realistic multi-line resume snippet — name line, contact
    line, EDUCATION header, university line, degree line, then 1-3 distractor
    lines drawn from skills/projects/job-titles/companies/certifications —
    so the model learns NAME/UNIVERSITY in the context they actually appear
    in, surrounded by exactly the kind of noise that fooled the old model.
    """
    name = rng.choice(names)
    uni = rng.choice(unis)
    degree = rng.choice(DEGREE_LINES)
    gpa = f"{rng.uniform(6.5, 9.8):.1f}"

    lines = [
        (name, "NAME"),
        (f"{rng.choice(negatives['location_fragments'])} | {rng.choice(negatives['contact_lines'])}", None),
        ("EDUCATION", None),
        (uni, "UNIVERSITY"),
        (f"{degree}, CGPA: {gpa}", None),
        (rng.choice(negatives["section_headers"]), None),
    ]
    filler_pool = (
        negatives["skill_lines"] + negatives["project_bullets"]
        + negatives["job_titles"] + negatives["company_names"]
        + negatives["certifications"]
    )
    for _ in range(rng.randint(1, 3)):
        lines.append((rng.choice(filler_pool), None))

    return _build_text(lines)


def generate_negative_example(negatives, rng):
    """Pure noise: 1-2 lines guaranteed to contain zero entities."""
    all_lines = [line for pool in negatives.values() for line in pool]
    chosen = [rng.choice(all_lines) for _ in range(rng.randint(1, 2))]
    return "\n".join(chosen), []


def generate_dataset(names, unis, negatives, count, rng):
    examples = []
    for _ in range(count):
        roll = rng.random()
        if roll < 0.35:
            examples.append(generate_sentence_example(names, unis, rng))
        elif roll < 0.70:
            examples.append(generate_mini_resume_example(names, unis, negatives, rng))
        else:
            examples.append(generate_negative_example(negatives, rng))
    return examples


def to_docbin(examples, nlp):
    db = DocBin()
    skipped = 0
    for text, entities in examples:
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue
            spans.append(span)
        try:
            doc.ents = spans
        except ValueError:
            skipped += 1
            continue
        db.add(doc)
    return db, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate expanded synthetic NER training data.")
    parser.add_argument("--train-count", type=int, default=1500)
    parser.add_argument("--dev-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Hold out a slice of names/universities/negatives entirely for dev, so
    # dev.spacy tests generalization to unseen entity values instead of
    # re-testing memorized training examples.
    train_names, dev_names = _split_pool(NAMES, 0.8, rng)
    train_unis, dev_unis = _split_pool(UNIVERSITIES, 0.8, rng)
    train_negatives, dev_negatives = {}, {}
    for key, pool in NEGATIVE_POOLS.items():
        tr, dv = _split_pool(pool, 0.8, rng)
        train_negatives[key] = tr
        dev_negatives[key] = dv

    train_examples = generate_dataset(train_names, train_unis, train_negatives, args.train_count, rng)
    dev_examples = generate_dataset(dev_names, dev_unis, dev_negatives, args.dev_count, rng)

    nlp = spacy.blank("en")
    train_db, train_skipped = to_docbin(train_examples, nlp)
    dev_db, dev_skipped = to_docbin(dev_examples, nlp)

    train_db.to_disk("train.spacy")
    dev_db.to_disk("dev.spacy")

    print(f"train.spacy: {len(train_examples)} examples ({train_skipped} entity spans skipped) "
          f"-- {len(train_names)} names / {len(train_unis)} universities")
    print(f"dev.spacy:   {len(dev_examples)} examples ({dev_skipped} entity spans skipped) "
          f"-- {len(dev_names)} names / {len(dev_unis)} universities (all held out from train)")


if __name__ == "__main__":
    main()
