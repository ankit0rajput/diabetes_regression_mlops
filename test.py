import requests

url = "https://my-model-endpoint1.eastus2.inference.ml.azure.com/score"
# key = ""

headers = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {key}"
}

data = {
    "data": [[0.05, 0.03, 0.02, 0.01, 0.04, 0.02, 0.03, 0.01, 0.02, 0.05]]
}

response = requests.post(url, json=data, headers=headers)

print(response.json())




# curl -X POST https://my-model-endpoint1.eastus2.inference.ml.azure.com/score \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer 6ae15bgyO5mdosH7o7gDER3MfPMH4tHgtDhQCVBwc8IVsXhxG1RdJQQJ99CBAAAAAAAAAAAAINFRAZML3k1T" \
#   -d '{"data": [0.05, 0.03, 0.02, 0.01, 0.04, 0.02, 0.03, 0.01, 0.02, 0.05]}'