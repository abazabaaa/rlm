"""
Example: Multi-Document Semantic Reasoning
This demonstrates RLM's ability to answer questions requiring information across multiple documents.
"""
from rlm.rlm_repl import RLM_REPL

def generate_product_catalog():
    """Generate a multi-document product catalog with cross-references."""
    documents = []

    # Generate 100 product descriptions with varying attributes
    products = [
        {"id": "PROD-001", "name": "Wireless Mouse X1000", "category": "Electronics", "price": 49.99, "rating": 4.5, "manufacturer": "TechCorp", "year": 2023},
        {"id": "PROD-002", "name": "USB-C Adapter Pro", "category": "Accessories", "price": 29.99, "rating": 4.2, "manufacturer": "ConnectCo", "year": 2022},
        {"id": "PROD-003", "name": "Mechanical Keyboard Elite", "category": "Electronics", "price": 149.99, "rating": 4.8, "manufacturer": "TechCorp", "year": 2024},
        {"id": "PROD-004", "name": "Laptop Stand Deluxe", "category": "Accessories", "price": 79.99, "rating": 4.6, "manufacturer": "ErgoTech", "year": 2023},
        {"id": "PROD-005", "name": "Webcam HD Pro", "category": "Electronics", "price": 89.99, "rating": 4.3, "manufacturer": "VisionCorp", "year": 2023},
    ]

    # Add reviews for each product (separate documents)
    reviews = [
        {"product_id": "PROD-001", "user": "john_d", "rating": 5, "text": "Amazing mouse! The ergonomics are perfect and battery lasts forever."},
        {"product_id": "PROD-001", "user": "sarah_m", "rating": 4, "text": "Good mouse but a bit pricey."},
        {"product_id": "PROD-003", "user": "tech_guru", "rating": 5, "text": "Best keyboard I've ever used. The TechCorp products are always top-notch."},
        {"product_id": "PROD-003", "user": "gamer_pro", "rating": 5, "text": "Perfect for gaming and typing. Worth every penny!"},
        {"product_id": "PROD-004", "user": "office_worker", "rating": 5, "text": "Improved my posture significantly. Highly recommend!"},
    ]

    # Add manufacturer information (separate documents)
    manufacturers = [
        {"name": "TechCorp", "founded": 2010, "headquarters": "San Francisco", "specialty": "High-end electronics", "awards": ["Best Innovation 2023", "Consumer Choice 2024"]},
        {"name": "ConnectCo", "founded": 2015, "headquarters": "Austin", "specialty": "Connectivity solutions", "awards": []},
        {"name": "ErgoTech", "founded": 2018, "headquarters": "Seattle", "specialty": "Ergonomic workspace products", "awards": ["Ergonomics Award 2023"]},
        {"name": "VisionCorp", "founded": 2012, "headquarters": "Boston", "specialty": "Imaging devices", "awards": ["Best Camera Tech 2022"]},
    ]

    # Combine into a large context
    context = "# PRODUCT CATALOG DATABASE\n\n"

    context += "## PRODUCTS\n"
    for p in products:
        context += f"\n### {p['id']}\n"
        context += f"Name: {p['name']}\n"
        context += f"Category: {p['category']}\n"
        context += f"Price: ${p['price']}\n"
        context += f"Rating: {p['rating']}/5\n"
        context += f"Manufacturer: {p['manufacturer']}\n"
        context += f"Year: {p['year']}\n"

    context += "\n## CUSTOMER REVIEWS\n"
    for r in reviews:
        context += f"\nProduct: {r['product_id']}\n"
        context += f"User: {r['user']}\n"
        context += f"Rating: {r['rating']}/5\n"
        context += f"Review: {r['text']}\n"

    context += "\n## MANUFACTURERS\n"
    for m in manufacturers:
        context += f"\n### {m['name']}\n"
        context += f"Founded: {m['founded']}\n"
        context += f"Headquarters: {m['headquarters']}\n"
        context += f"Specialty: {m['specialty']}\n"
        context += f"Awards: {', '.join(m['awards']) if m['awards'] else 'None'}\n"

    return context

def main():
    print("Example: Multi-Document Semantic Reasoning")
    print("=" * 80)

    context = generate_product_catalog()
    print(f"Context size: {len(context)} characters\n")

    # Complex query requiring multi-hop reasoning across documents
    query = """
    Find all products made by manufacturers that have won awards in 2023 or later,
    and have an average customer rating above 4.5 stars based on the reviews.
    List the product names and explain why they qualify.
    """

    print(f"Query: {query}\n")
    print("Running RLM with recursive model...\n")

    rlm = RLM_REPL(
        model="gpt-5-nano",
        recursive_model="gpt-5",
        enable_logging=True,
        max_iterations=15
    )

    result = rlm.completion(context=context, query=query)
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result)

if __name__ == "__main__":
    main()
