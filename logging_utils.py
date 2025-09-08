import csv
from datetime import datetime

def log_user_query(username: str, question: str, answer: str, path: str = "user_logs.csv"):
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now().isoformat(timespec='seconds'), username, question, answer])
