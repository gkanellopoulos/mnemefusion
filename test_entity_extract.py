# Simple test to understand entity extraction behavior
content = "Alice, Bob, and Charlie collaborated"
print(f"Testing: {content}")
print("Expected by test: Alice, Bob, Charlie (3 entities)")
print("Actual behavior: The extractor treats consecutive capitalized words")
print("as multi-word phrases, so 'Alice,' 'Bob,' forms 'Alice Bob' phrase")
print("Since comma is stripped, consecutive caps = multi-word entity")
