"""
Example: Structured Data Analysis with Large Datasets
This demonstrates RLM's ability to filter and aggregate structured data programmatically.
Similar to the OOLONG benchmark mentioned in the blog post.
"""
from rlm.rlm_repl import RLM_REPL
import random
import json

def generate_user_activity_log(num_entries=5000):
    """Generate a large user activity log with structured data."""

    users = [f"user_{i:04d}" for i in range(1, 101)]  # 100 users
    actions = ["login", "logout", "view_page", "click_button", "submit_form", "download", "upload", "search", "edit_profile"]
    pages = ["home", "dashboard", "settings", "profile", "reports", "analytics", "help", "admin"]

    entries = []
    for i in range(num_entries):
        entry = {
            "id": i + 1,
            "timestamp": f"2024-10-{random.randint(1, 30):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
            "user_id": random.choice(users),
            "action": random.choice(actions),
            "page": random.choice(pages),
            "duration_seconds": random.randint(1, 300),
            "success": random.choice([True, True, True, False]),  # 75% success rate
        }
        entries.append(entry)

    # Convert to JSON string
    return json.dumps(entries, indent=2)

def main():
    print("Example: Structured Data Analysis")
    print("=" * 80)

    # Generate a large activity log
    num_entries = 5000
    context = generate_user_activity_log(num_entries)
    print(f"Generated {num_entries} activity log entries")
    print(f"Context size: {len(context)} characters (~{len(context)/1000:.1f}KB)\n")

    # Complex analytical query
    query = """
    Analyze this user activity log and provide the following:
    1. Identify the top 5 users with the most failed actions
    2. For each of these users, calculate:
       - Total number of actions
       - Failure rate (percentage)
       - Most common failed action type
       - Most common page where failures occur
    3. Provide a summary explaining any patterns you notice

    Format your answer as a structured report.
    """

    print(f"Query: {query}\n")
    print("Running RLM with recursive model...\n")

    rlm = RLM_REPL(
        model="gpt-5-nano",
        recursive_model="gpt-5",
        enable_logging=True,
        max_iterations=20
    )

    result = rlm.completion(context=context, query=query)
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result)

if __name__ == "__main__":
    main()
