# support_docs_loader.py
# Replace this with your real document loading code

def load_support_documents():
    # Mock example. Replace with actual loading logic.
    # Each doc: {id, content, category, priority, date}
    docs = [
        {
            "id": "doc001",
            "content": "How do I reset my password? You can reset your password by clicking the 'Forgot Password' link on the login page...",
            "category": "account",
            "priority": "normal",
            "date": "2024-06-10"
        },
        {
            "id": "doc002",
            "content": "Payment failed due to insufficient funds. Please check your card information or try a different payment method...",
            "category": "billing",
            "priority": "urgent",
            "date": "2024-06-07"
        },
        # ... Load up to 8,000 real docs in production
    ]
    return docs
