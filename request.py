import requests

# Replace with the Gemini API URL
api_url = "http://localhost:5000/generate"

# Replace with your API key if authentication is required
headers = {
    "Content-Type": "application/json",
}

prompt="// Focal class\npublic class Query {\nprotected String serverURL,\n    associatesID,\n    token = \"DSB0XDDW1GQ3S\", //don't change A4J won't work without this. Used for tracking tool use.\n    searchType,\n    type,\n    page,\n    offer;\nprotected ArrayList searchValues;\na4jUtil jawsUtil = new a4jUtil();\npublic Query(){}\n// Focal method\npublic String queryGenerator(String searchType, String type, String page, String offer, ArrayList searchValues) {\n        //    log.debug(\"queryGenerator - in\");\n        StringBuffer buffer = new StringBuffer();\n        buffer.append(serverURL);\n        buffer.append(\"?\");\n        buffer.append(\"t=\");\n        buffer.append(associatesID);\n        buffer.append(\"&\");\n        buffer.append(\"dev-t=\");\n        buffer.append(token);\n        buffer.append(\"&\");\n        buffer.append(searchType);\n        buffer.append(\"=\");\n        buffer.append(generateMultipleSearchString(searchType, searchValues));\n        buffer.append(\"&\");\n        buffer.append(\"type=\");\n        buffer.append(type);\n        buffer.append(\"&\");\n        buffer.append(\"offerpage=\");\n        buffer.append(page);\n        buffer.append(\"&\");\n        buffer.append(\"offer=\");\n        buffer.append(offer);\n        buffer.append(\"&\");\n        buffer.append(\"f=xml\");\n        //      log.debug(\"queryGenerator - out\");\n        return new String(buffer);\n    }\n}\nPlease infer the intention of the \"queryGenerator(String, String, String, String, ArrayList)\"."

# Payload for the request
payload = {
  "contents": [
    {
      "parts": [
        {
          "text": prompt
        }
      ]
    }
  ],
  "maxTokens": 1024
}

payload = {
  "model":"meta-llama/Llama-3.2-3B-Instruct",
  "prompt":prompt
}

# Send the POST request
response = requests.post(api_url, headers=headers, json=payload)

# Check the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Failed with status code {response.status_code}: {response.text}")

