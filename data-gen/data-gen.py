import csv
import random
from faker import Faker
from tqdm import tqdm

# Initialize Faker
fake = Faker()

output_file = "synthetic_transactions.csv"
fieldnames = [
    'user_id', 'age', 'region', 'credit_score', 'behavioral_history',
    'timestamp', 'amount', 'merchant', 'ip', 'device_fingerprint',
    'location', 'velocity', 'browser', 'network_latency', 'session_metadata', 'label'
]

with open(output_file, 'w', newline='', buffering=1) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for _ in tqdm(range(100_000_000), desc="Generating records"):
        record = {
            'user_id': fake.uuid4(),
            'age': random.randint(18, 80),
            'region': fake.country(),
            'credit_score': random.randint(300, 850),
            'behavioral_history': random.choice(['normal', 'suspicious', 'fraudulent']),
            'timestamp': fake.date_time_this_year(),
            'amount': round(random.uniform(10, 10000), 2),
            'merchant': fake.company(),
            'ip': fake.ipv4(),
            'device_fingerprint': fake.sha256(),
            'location': fake.city(),
            'velocity': round(random.uniform(0, 10), 2),
            'browser': fake.user_agent(),
            'network_latency': random.randint(10, 500),
            'session_metadata': fake.sentence(nb_words=6),
            'label': random.choices(['normal', 'suspicious', 'fraudulent'], weights=[0.9, 0.08, 0.02])[0]
        }
        writer.writerow(record)
        print(f"Record {_} generated.")

print("Finished generating 100M records.")