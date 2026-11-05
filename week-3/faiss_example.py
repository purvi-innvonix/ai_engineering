
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# Sample documents
documents = [
    """
    Refund Policy: Customers can request a refund within 30 days of
    receiving a product. The product must be unused, in its original
    packaging, and accompanied by proof of purchase. After the returned
    product passes inspection, the money is sent to the original
    payment method.
    """,

    """
    Credit Card Refunds: Once a refund is approved, credit card payments
    usually take five to seven business days to appear in the customer's
    bank account. The store may mark the refund as processed before
    the bank displays the money in the transaction history.
    """,

    """
    Return Process: Customers should open their order history, select
    the purchased product, choose Request a Return, and explain the
    reason. After approval, they receive instructions for packaging
    and shipping the item back to the warehouse.
    """,

    """
    Damaged Products: If an item arrives broken, customers should report
    the problem within 48 hours. They should provide photographs of
    the damaged product and its packaging. Depending on the investigation,
    the company may offer a replacement or return the payment.
    """,

    """
    Failed Payments: Sometimes a bank deducts money even though the
    order page says payment failed. This can be a temporary authorization.
    Customers should check whether the transaction is pending or completed
    before contacting billing support.
    """,

    """
    Duplicate Transactions: If a customer sees two completed charges
    for one purchase, they should send both transaction references,
    the order number, and the transaction dates to the billing team.
    The team checks payment records before approving a correction.
    """,

    """
    Delivery Tracking: Standard delivery takes three to five business
    days after dispatch. Tracking information becomes available when
    the courier scans the parcel. Remote locations, bad weather,
    and public holidays can cause additional delays.
    """,

    """
    Missing Packages: If tracking says delivered but the customer
    cannot find the parcel, they should check reception, building
    security, nearby delivery locations, and household members.
    If the parcel is still missing, support can investigate with
    the courier using the tracking number.
    """,

    """
    Cancellation Policy: Orders can usually be cancelled before
    warehouse packing begins. Once a package has been dispatched,
    cancellation may no longer be possible. Customers may need to
    receive the package and follow the normal return procedure.
    """,

    """
    Digital Purchases: Downloadable products are generally not
    refundable after downloading has started. However, customers
    experiencing a failed download or an accidental purchase can
    contact support for an individual review of their situation.
    """,

    """
    Wallet Payments: Refunds for purchases made through a digital
    wallet are returned to the same wallet. Some wallet services
    show the money immediately, while others need several business
    days to update the customer's available balance.
    """,

    """
    Subscription Billing: Subscriptions renew automatically unless
    cancelled before the renewal date. Cancelling stops future
    payments but does not normally return the fee for the current
    billing period. Customers can check their renewal date online.
    """,

    """
    Warranty Claims: Manufacturing defects may be covered during
    the warranty period. Accidental damage and unauthorized repairs
    are usually excluded. Customers need the serial number and
    proof of purchase to submit a warranty request.
    """,

    """
    Product Exchanges: Eligible products can sometimes be exchanged
    for another size or colour. If the replacement costs more,
    the customer pays the difference. If it costs less, the remaining
    amount is handled according to the store's refund policy.
    """,

    """
    Discount Problems: A promotional code may fail because it has
    expired, applies only to selected products, or requires a minimum
    purchase. Customers should review the terms before asking the
    pricing team to investigate a discount that was not applied.
    """,

    """
    Account Security: Customers who notice an unfamiliar transaction
    should check whether another authorized user made the purchase.
    If fraud is suspected, they should contact their bank promptly
    and notify the store's billing team about the suspicious activity.
    """,

    """
    Support Priorities: Payment failures that prevent a customer
    from completing checkout may be more urgent than general product
    questions. Customers should include relevant order numbers,
    error messages, and transaction details when contacting support.
    """,

    """
    Multiple Shipments: An order containing several products may
    arrive in separate packages from different warehouses. Each
    package can have its own tracking number and delivery date.
    Customers should check all shipment records before reporting
    that a product has not arrived.
    """,

    """
    Refund Status: Requested means the customer has submitted a
    refund request. Approved means the store accepted it. Processed
    means the payment has been sent to the payment provider, but
    the bank may still need time to show the money in the account.
    """,

    """
    Refund Rejection: A return may be rejected when it is outside
    the return period, the product shows signs of use, accessories
    are missing, or the returned item does not match the purchase.
    Customers can contact support to ask for a review of the decision.
    """
]

# Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

query = "How can I get my money back?"

# ==================================
# 1. WITHOUT FAISS
# ==================================

print("\nWITHOUT FAISS")

# Convert documents and question into numbers
doc_vectors = np.array(
    embeddings.embed_documents(documents)
)

query_vector = np.array(
    embeddings.embed_query(query)
)

# Compare question with every document
scores = (
    doc_vectors @ query_vector
    / (
        np.linalg.norm(doc_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )
)

# Select the two most similar documents
top_indices = np.argsort(scores)[::-1][:2]

for index in top_indices:
    print(documents[index])
    print("Similarity:", round(float(scores[index]), 4))


# ==================================
# 2. WITH FAISS
# ==================================

print("\nWITH FAISS")

# Create the search index
vector_store = FAISS.from_texts(
    documents,
    embedding=embeddings
)

# Search for the two most similar documents
results = vector_store.similarity_search(
    query,
    k=2
)

for document in results:
    print(document.page_content)
