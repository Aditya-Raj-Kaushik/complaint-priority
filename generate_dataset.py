import random
import pandas as pd

random.seed(42)

ISSUES = [
    "login", "payment", "dashboard", "profile update",
    "data export", "report generation", "search",
    "notifications", "API access", "file upload"
]

SYMPTOMS = [
    "fails intermittently",
    "is slower than expected",
    "sometimes does not respond",
    "shows unexpected behavior",
    "returns inconsistent results",
    "works only after retrying",
    "causes confusion for users",
]

CONTEXTS = [
    "during peak hours",
    "after recent update",
    "on mobile devices",
    "for certain users",
    "under heavy load",
    "when switching accounts",
    "after long inactivity",
]

EMOTIONS = [
    "This is frustrating",
    "Quite disappointed",
    "Annoying experience",
    "Not sure why this happens",
    "This impacts my work",
    "Hard to rely on the system",
]

REQUESTS = [
    "please look into this",
    "needs investigation",
    "would appreciate a fix",
    "hope this can be resolved",
    "requesting clarification",
    "seeking assistance",
]

def generate_text():
    issue = random.choice(ISSUES)
    symptom = random.choice(SYMPTOMS)
    context = random.choice(CONTEXTS)
    emotion = random.choice(EMOTIONS)
    request = random.choice(REQUESTS)

    patterns = [
        f"{issue} {symptom} {context}. {emotion}, {request}.",
        f"Users report that {issue} {symptom} {context}. {request}.",
        f"{emotion}. The {issue} {symptom} {context}.",
        f"Observed that {issue} {symptom} {context}, {request}.",
    ]

    if random.random() < 0.35:
        issue2 = random.choice([i for i in ISSUES if i != issue])
        patterns.append(
            f"{issue} {symptom} and {issue2} also {random.choice(SYMPTOMS)} {context}. {emotion}."
        )

    return random.choice(patterns)

def assign_priority(text: str):
    """
    Priority depends on *latent factors*, not keywords.
    """
    base = random.random()

    if "intermittently" in text or "retrying" in text:
        base += 0.15
    if "under heavy load" in text or "peak hours" in text:
        base += 0.15
    if "multiple issues" in text:
        base += 0.10

    base += random.uniform(-0.15, 0.15)

    if base > 0.65:
        return "high"
    elif base > 0.35:
        return "medium"
    else:
        return "low"


def generate_row():
    text = generate_text()
    priority = assign_priority(text)

    return {
        "complaint_text": text,
        "customer_tenure": random.randint(0, 15),
        "priority": priority,
    }


def main():
    rows = [generate_row() for _ in range(8000)]

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/complaints.csv", index=False)

    print("Dataset generated:", df.shape)
    print(df["priority"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
