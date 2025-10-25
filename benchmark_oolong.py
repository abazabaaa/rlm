"""
OOLONG-Style Benchmark for RLM
Based on: https://openreview.net/forum?id=lrDr6dmXOX

This creates a synthetic version of the OOLONG benchmark which tests:
- Long-context reasoning over fine-grained information
- Semantic mapping and association of thousands of pieces of information
- Classification and aggregation across large datasets

The task requires models to:
1. Read thousands of entries (Date, User ID, Question instances)
2. Infer implicit labels for questions (e.g., "entity", "description", "numeric")
3. Filter by specific user IDs
4. Count how many match certain criteria

Performance: GPT-5 achieves <50% accuracy on the original OOLONG at 128K context.
"""

from rlm.rlm_repl import RLM_REPL
import random
import json
from datetime import datetime, timedelta

# Question templates for different categories
ENTITY_QUESTIONS = [
    "Who is {name}?",
    "What is {thing}?",
    "Who was {name}?",
    "What are {things}?",
]

DESCRIPTION_QUESTIONS = [
    "How does {thing} work?",
    "What does {thing} do?",
    "How do you use {tool}?",
    "What is the purpose of {thing}?",
]

NUMERIC_QUESTIONS = [
    "How many {things} are there?",
    "How old is {name}?",
    "How many years has it been since {event}?",
    "What is the population of {place}?",
]

TEMPORAL_QUESTIONS = [
    "When did {event} happen?",
    "What year was {name} born?",
    "When was {thing} invented?",
    "What date did {event} occur?",
]

# Sample data for filling templates
NAMES = ["John Smith", "Marie Curie", "Albert Einstein", "Rosa Parks", "Nelson Mandela"]
THINGS = ["photosynthesis", "quantum mechanics", "the internet", "democracy", "evolution"]
TOOLS = ["a microscope", "a telescope", "a computer", "a thermometer"]
EVENTS = ["World War II", "the moon landing", "the Renaissance", "the Industrial Revolution"]
PLACES = ["Tokyo", "New York City", "London", "Mumbai", "São Paulo"]

def generate_question(category):
    """Generate a question for a specific category."""
    if category == "entity":
        template = random.choice(ENTITY_QUESTIONS)
        return template.format(
            name=random.choice(NAMES),
            thing=random.choice(THINGS),
            things=random.choice(THINGS)
        )
    elif category == "description":
        template = random.choice(DESCRIPTION_QUESTIONS)
        return template.format(
            thing=random.choice(THINGS),
            tool=random.choice(TOOLS)
        )
    elif category == "numeric":
        template = random.choice(NUMERIC_QUESTIONS)
        return template.format(
            things=random.choice(THINGS),
            name=random.choice(NAMES),
            event=random.choice(EVENTS),
            place=random.choice(PLACES)
        )
    else:  # temporal
        template = random.choice(TEMPORAL_QUESTIONS)
        return template.format(
            event=random.choice(EVENTS),
            name=random.choice(NAMES),
            thing=random.choice(THINGS)
        )

def generate_oolong_dataset(num_entries=5000):
    """Generate synthetic OOLONG-style dataset."""
    categories = ["entity", "description", "numeric", "temporal"]

    # Generate entries
    entries = []
    start_date = datetime(2020, 1, 1)

    for i in range(num_entries):
        # Random date
        random_days = random.randint(0, 1095)  # ~3 years
        date = start_date + timedelta(days=random_days)
        date_str = date.strftime("%b %d, %Y")

        # Random user ID (100 users)
        user_id = random.randint(1, 100)

        # Random category and question
        category = random.choice(categories)
        question = generate_question(category)

        entry = {
            "date": date_str,
            "user_id": user_id,
            "question": question,
            "category": category  # Ground truth (hidden from model)
        }
        entries.append(entry)

    return entries

def format_dataset_for_context(entries, include_labels=False):
    """Format entries as context string (mimicking OOLONG format)."""
    lines = []
    for entry in entries:
        line = f"Date: {entry['date']} || User: {entry['user_id']} || Instance: {entry['question']}"
        if include_labels:
            line += f" || Label: {entry['category']}"
        lines.append(line)
    return "\n".join(lines)

def create_query(target_users, target_category):
    """Create an OOLONG-style query."""
    user_list = ", ".join(map(str, target_users))
    query = f"""
Analyze the provided dataset of user question instances.

Your task:
Among instances associated with users [{user_list}], count how many should be classified
as category '{target_category}'.

To solve this:
1. Parse all entries and identify which users are in the target set
2. For each instance from these users, infer the implicit category:
   - 'entity': Questions asking "who" or "what is" something/someone
   - 'description': Questions asking "how" something works or functions
   - 'numeric': Questions asking "how many" or numerical quantities
   - 'temporal': Questions asking "when" or about dates/time

3. Count how many instances match the target category '{target_category}'
4. Return ONLY the count as an integer

Note: You must infer the category semantically - no labels are provided in the dataset.
Use the REPL environment and potentially recursive LLM calls to chunk and process this data.
"""
    return query

def main():
    print("=" * 80)
    print("OOLONG-STYLE BENCHMARK FOR RLM")
    print("=" * 80)
    print("\nGenerating synthetic dataset (OOLONG-style)...")

    # Generate dataset
    entries = generate_oolong_dataset(num_entries=5000)

    # Format as context (no labels exposed)
    context = format_dataset_for_context(entries, include_labels=False)
    print(f"Generated {len(entries)} entries")
    print(f"Context size: {len(context)} characters (~{len(context)/1000:.1f}KB)")

    # Create task: find all "entity" questions from users 5, 12, 23, 45, 67
    target_users = [5, 12, 23, 45, 67]
    target_category = "entity"

    # Calculate ground truth
    ground_truth = sum(
        1 for e in entries
        if e['user_id'] in target_users and e['category'] == target_category
    )

    print(f"\nTask: Count '{target_category}' instances from users {target_users}")
    print(f"Ground truth answer: {ground_truth}")
    print(f"\nRunning RLM (this may take several minutes)...")
    print("=" * 80)

    # Create query
    query = create_query(target_users, target_category)

    # Run RLM
    rlm = RLM_REPL(
        model="gpt-5-nano",
        recursive_model="gpt-5",
        enable_logging=True,
        max_iterations=25
    )

    result = rlm.completion(context=context, query=query)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"RLM Answer: {result}")
    print(f"Ground Truth: {ground_truth}")

    # Try to parse result as integer
    try:
        result_int = int(result.strip())
        accuracy = "CORRECT" if result_int == ground_truth else "INCORRECT"
        error = abs(result_int - ground_truth)
        print(f"Accuracy: {accuracy}")
        if error > 0:
            print(f"Error: {error} off from ground truth")
    except ValueError:
        print(f"Could not parse result as integer: {result}")

    print("\n" + "=" * 80)
    print("BENCHMARK STATS")
    print("=" * 80)
    print(f"Dataset size: {len(entries)} entries")
    print(f"Number of users: 100")
    print(f"Target users: {len(target_users)}")
    print(f"Target category: {target_category}")
    print(f"Relevant entries: {sum(1 for e in entries if e['user_id'] in target_users)}")
    print(f"Matching entries (ground truth): {ground_truth}")

if __name__ == "__main__":
    main()
