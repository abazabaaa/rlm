"""
BrowseComp-Plus Style Benchmark for RLM (Mini Version)
Based on: https://arxiv.org/abs/2508.06600

This creates a mini version of BrowseComp-Plus which tests:
- Multi-hop reasoning across multiple documents
- Finding information scattered across different sources
- Synthesizing answers from distributed knowledge

The full BrowseComp-Plus has ~100K documents. This mini version uses 100 documents
to demonstrate the concept while remaining tractable.

Performance on full benchmark:
- GPT-5 with BM25: 55.9% accuracy
- GPT-5 with Qwen3-Embedding: 70.1% accuracy
- RLM should improve upon baseline GPT-5
"""

from rlm.rlm_repl import RLM_REPL
import random

# Document templates for different types
PRODUCT_DOCS = [
    """Product ID: {pid}
Name: {name}
Category: {category}
Manufacturer: {manufacturer}
Release Year: {year}
Price: ${price}
Description: {description}""",
]

REVIEW_DOCS = [
    """Review for Product {pid}
Reviewer: {reviewer}
Rating: {rating}/5
Date: {date}
Review: {review_text}""",
]

MANUFACTURER_DOCS = [
    """Manufacturer: {name}
Founded: {founded}
Headquarters: {headquarters}
Specialty: {specialty}
Notable Products: {products}
Awards: {awards}""",
]

def generate_documents(num_docs=100):
    """Generate synthetic documents similar to BrowseComp-Plus corpus."""
    documents = []

    # Generate some manufacturers
    manufacturers = [
        {"name": "TechCorp", "founded": 2010, "hq": "San Francisco", "specialty": "Electronics", "awards": ["Innovation Award 2023"]},
        {"name": "GadgetPro", "founded": 2015, "hq": "Austin", "specialty": "Smart Devices", "awards": ["Best Design 2024"]},
        {"name": "DeviceMakers", "founded": 2018, "hq": "Seattle", "specialty": "IoT Products", "awards": []},
    ]

    # Generate manufacturer docs
    for mfg in manufacturers:
        doc = f"""Manufacturer: {mfg['name']}
Founded: {mfg['founded']}
Headquarters: {mfg['hq']}
Specialty: {mfg['specialty']}
Awards: {', '.join(mfg['awards']) if mfg['awards'] else 'None'}"""
        documents.append({"id": f"MFG-{mfg['name']}", "content": doc, "type": "manufacturer"})

    # Generate product docs
    products = []
    for i in range(40):
        mfg = random.choice(manufacturers)
        product = {
            "pid": f"PROD-{i+1:03d}",
            "name": f"{random.choice(['Smart', 'Pro', 'Ultra', 'Elite'])} {random.choice(['Watch', 'Phone', 'Tablet', 'Laptop'])} {i+1}",
            "category": random.choice(["Electronics", "Accessories", "Wearables"]),
            "manufacturer": mfg["name"],
            "year": random.randint(2020, 2024),
            "price": random.randint(50, 2000),
            "rating": round(random.uniform(3.5, 5.0), 1)
        }
        products.append(product)

        doc = f"""Product ID: {product['pid']}
Name: {product['name']}
Category: {product['category']}
Manufacturer: {product['manufacturer']}
Release Year: {product['year']}
Price: ${product['price']}
Average Rating: {product['rating']}/5.0"""
        documents.append({"id": product["pid"], "content": doc, "type": "product"})

    # Generate reviews (scattered across documents)
    for i in range(40):
        product = random.choice(products)
        doc = f"""Product Review
Product ID: {product['pid']}
Product Name: {product['name']}
Reviewer: User{random.randint(1, 100)}
Rating: {random.randint(3, 5)}/5
Date: 2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}
Review: {random.choice([
    'Great product! Highly recommend.',
    'Good quality but a bit pricey.',
    'Excellent build quality and features.',
    'Works as expected, very satisfied.',
    'Amazing device, worth every penny.'
])}"""
        documents.append({"id": f"REVIEW-{i+1:03d}", "content": doc, "type": "review"})

    # Add some technical spec docs
    for i in range(17):
        product = random.choice(products)
        doc = f"""Technical Specifications
Product: {product['name']} ({product['pid']})
Battery Life: {random.randint(10, 72)} hours
Weight: {random.randint(100, 1000)}g
Connectivity: Bluetooth {random.choice(['4.2', '5.0', '5.2'])}, WiFi {random.choice(['5', '6', '6E'])}
Water Resistance: {random.choice(['IP67', 'IP68', 'None'])}
Warranty: {random.choice(['1 year', '2 years', '3 years'])}"""
        documents.append({"id": f"SPEC-{i+1:03d}", "content": doc, "type": "specs"})

    return documents, products, manufacturers

def format_corpus(documents):
    """Format documents as a corpus string."""
    corpus = []
    for doc in documents:
        corpus.append(f"=== DOCUMENT {doc['id']} ===")
        corpus.append(doc['content'])
        corpus.append("")  # blank line
    return "\n".join(corpus)

def create_multi_hop_query(products, manufacturers):
    """Create a multi-hop query requiring information from multiple documents."""

    # Find a product from an award-winning manufacturer with high rating
    candidates = [
        p for p in products
        if any(m['name'] == p['manufacturer'] and m['awards'] for m in manufacturers)
        and p['rating'] >= 4.5
    ]

    if not candidates:
        # Fallback query
        return """
Find all products that meet ALL of the following criteria:
1. Manufactured by a company that has won awards
2. Released in 2023 or later
3. Have an average rating of 4.5 or higher

For each qualifying product, list:
- Product ID and name
- Manufacturer
- Why it qualifies (explain which awards the manufacturer won)

Then summarize: How many products match all criteria?
"""

    target = random.choice(candidates)
    target_mfg = next(m for m in manufacturers if m['name'] == target['manufacturer'])

    query = f"""
This is a multi-hop reasoning task requiring you to connect information across multiple documents.

Find a product that matches ALL of these criteria:
1. Manufactured by {target_mfg['name']}
2. Has won the award: {target_mfg['awards'][0] if target_mfg['awards'] else 'N/A'}
3. Has an average rating of 4.5 or higher
4. Released in {target['year']} or later

Your task:
1. First, verify which manufacturers have won awards by finding manufacturer documents
2. Then, find products made by those manufacturers
3. Filter products by rating and release year
4. List ALL products that match the criteria

Expected format:
Product ID: [ID]
Product Name: [Name]
Justification: [Explain why it matches all criteria]

Note: You should use chunking and the llm_query() function to process documents efficiently.
"""
    return query, target

def main():
    print("=" * 80)
    print("BROWSECOMP-PLUS STYLE BENCHMARK (MINI VERSION)")
    print("=" * 80)
    print("\nGenerating synthetic document corpus...")

    # Generate documents
    documents, products, manufacturers = generate_documents(num_docs=100)
    corpus = format_corpus(documents)

    print(f"Generated {len(documents)} documents")
    print(f"  - {sum(1 for d in documents if d['type'] == 'manufacturer')} manufacturer docs")
    print(f"  - {sum(1 for d in documents if d['type'] == 'product')} product docs")
    print(f"  - {sum(1 for d in documents if d['type'] == 'review')} review docs")
    print(f"  - {sum(1 for d in documents if d['type'] == 'specs')} technical spec docs")
    print(f"Corpus size: {len(corpus)} characters (~{len(corpus)/1000:.1f}KB)")

    # Create multi-hop query
    query, expected = create_multi_hop_query(products, manufacturers)

    print(f"\nThis is a multi-hop reasoning task:")
    print("The answer requires connecting information from:")
    print("  1. Manufacturer documents (to find award winners)")
    print("  2. Product documents (to find products by those manufacturers)")
    print("  3. Rating information (to filter by quality)")
    print("\nExpected product to find:")
    print(f"  {expected['name']} ({expected['pid']}) - {expected['manufacturer']}")
    print(f"  Rating: {expected['rating']}/5, Year: {expected['year']}")

    print(f"\nRunning RLM (this may take 1-2 minutes)...")
    print("=" * 80)

    # Run RLM
    rlm = RLM_REPL(
        model="gpt-5-nano",
        recursive_model="gpt-5",
        enable_logging=True,
        max_iterations=20
    )

    result = rlm.completion(context=corpus, query=query)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(result)

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    if expected['pid'] in result:
        print(f"✅ SUCCESS: Found expected product {expected['pid']}")
    else:
        print(f"❌ MISS: Did not find expected product {expected['pid']}")

    print(f"\nExpected: {expected['name']} ({expected['pid']})")
    print(f"Manufacturer: {expected['manufacturer']}")
    print(f"Rating: {expected['rating']}/5")

if __name__ == "__main__":
    main()
