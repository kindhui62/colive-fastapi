import requests

url = "http://127.0.0.1:8000/generate"

payload = {
    "user_input": "I think we need a clear rule about parties. It's been too loud some nights.",
    "history": [
        "Jamie: I didn’t sleep well last weekend — it was just too noisy.",
        "Alex: I was hosting a game night. But I get it, sorry if it went late."
    ],
    "avatars": ["Alice", "Benji", "Caden"],
    "participant_role": "Alice"
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("✅ GPT Response:")
    for turn in response.json()["dialogue"]:
        print(f'{turn["speaker"]} ({turn["emotion"]}, {turn["gesture"]}): {turn["text"]}')
else:
    print(f"❌ Error {response.status_code}:")
    print(response.text)
