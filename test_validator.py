from agent.validator import validate_image

result = validate_image(r"data\demo\test_image.jpg")

print("SatQuery AI Input Validator Test")
print("--------------------------------")
print("Valid:", result["valid"])
print("Message:", result["message"])