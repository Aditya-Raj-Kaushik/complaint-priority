import random
import pandas as pd # type: ignore

random.seed(42)

complaint_templates = {
    "high": [
        "App crashes when I try to {}",
        "Payment failed multiple times while {}",
        "System is completely unusable during {}",
        "Urgent issue affecting business: {}",
        "Service outage noticed while {}"
    ],
    "medium": [
        "Slow response when trying to {}",
        "Delay observed in {}",
        "Issue occurs occasionally during {}",
        "Minor bug noticed while {}",
        "Support took time to resolve {}"
    ],
    "low": [
        "Feature request related to {}",
        "UI improvement suggestion for {}",
        "General feedback about {}",
        "Would like enhancement in {}",
        "Cosmetic issue noticed in {}"
    ]
}

actions = [
    "logging in",
    "making a payment",
    "updating profile",
    "uploading documents",
    "using the dashboard",
    "accessing reports",
    "resetting password"
]

products = ["Mobile App", "Web App", "Payments", "Customer Service"]

rows = []

for _ in range(300):  
    priority = random.choices(
        ["low", "medium", "high"],
        weights=[0.5, 0.3, 0.2] 
    )[0]

    template = random.choice(complaint_templates[priority])
    action = random.choice(actions)

    complaint_text = template.format(action)
    product = random.choice(products)
    customer_tenure = random.randint(0, 10)

    rows.append({
        "complaint_text": complaint_text,
        "product": product,
        "customer_tenure": customer_tenure,
        "priority": priority
    })

df = pd.DataFrame(rows)
df.to_csv("data/raw/complaints.csv", index=False)

print("Dataset generated:", df.shape)
