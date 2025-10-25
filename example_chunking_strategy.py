"""
Example: Intelligent Chunking with Recursive Sub-LM Calls
This demonstrates RLM's ability to chunk large contexts and use recursive sub-LLMs for analysis.
Similar to the strategies described in the blog post.
"""
from rlm.rlm_repl import RLM_REPL
import random

def generate_research_papers(num_papers=50):
    """Generate fake research paper abstracts on various AI topics."""

    topics = [
        ("Machine Learning", ["neural networks", "deep learning", "supervised learning", "unsupervised learning", "reinforcement learning"]),
        ("Natural Language Processing", ["transformers", "language models", "tokenization", "embeddings", "attention mechanisms"]),
        ("Computer Vision", ["object detection", "image segmentation", "convolutional networks", "image classification", "generative models"]),
        ("Robotics", ["path planning", "sensor fusion", "manipulation", "locomotion", "control systems"]),
        ("AI Safety", ["alignment", "interpretability", "robustness", "adversarial examples", "value learning"]),
    ]

    authors_pool = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "EMNLP", "ACL", "AAAI", "IJCAI"]

    papers = []
    for i in range(num_papers):
        topic_name, keywords = random.choice(topics)

        # Generate paper
        paper = {
            "id": f"PAPER-{i+1:03d}",
            "title": f"Advances in {random.choice(keywords).title()}: A Novel Approach",
            "authors": [f"{random.choice(authors_pool)} et al."],
            "year": random.randint(2020, 2024),
            "venue": random.choice(venues),
            "topic": topic_name,
            "citations": random.randint(0, 500),
            "abstract": f"This paper presents a novel approach to {random.choice(keywords)} in the field of {topic_name}. "
                       f"We demonstrate that our method outperforms existing baselines by {random.randint(5, 30)}% "
                       f"on standard benchmarks. Our key contribution is the integration of {random.choice(keywords)} "
                       f"with {random.choice(keywords)}, which enables more efficient processing. "
                       f"Experimental results show significant improvements in both accuracy and computational efficiency."
        }
        papers.append(paper)

    # Format as text
    context = "# RESEARCH PAPER DATABASE\n\n"
    context += f"Total Papers: {num_papers}\n\n"

    for paper in papers:
        context += f"## {paper['id']}: {paper['title']}\n"
        context += f"Authors: {', '.join(paper['authors'])}\n"
        context += f"Year: {paper['year']}\n"
        context += f"Venue: {paper['venue']}\n"
        context += f"Topic: {paper['topic']}\n"
        context += f"Citations: {paper['citations']}\n"
        context += f"Abstract: {paper['abstract']}\n"
        context += "\n" + "-" * 80 + "\n\n"

    return context

def main():
    print("Example: Intelligent Chunking with Recursive Sub-LMs")
    print("=" * 80)

    context = generate_research_papers(num_papers=50)
    print(f"Generated 50 research papers")
    print(f"Context size: {len(context)} characters (~{len(context)/1000:.1f}KB)\n")

    # Query that requires semantic understanding across all papers
    query = """
    I'm looking for highly influential papers (>200 citations) on topics related to
    language models or transformers that were published in top-tier venues (NeurIPS, ICML, ICLR).

    For each paper you find:
    1. List the paper ID and title
    2. Explain why it matches the criteria
    3. Summarize the key contribution

    Then provide a brief synthesis of the common themes across these papers.

    Note: You should use chunking and recursive sub-LLM calls to process this efficiently.
    """

    print(f"Query: {query}\n")
    print("Running RLM with recursive model...\n")
    print("Expected behavior: The model should chunk the papers and use llm_query() for semantic analysis\n")

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
