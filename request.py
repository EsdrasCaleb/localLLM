import requests

def check_response(response):
  # Check the response
  if response.status_code == 200:
      print("Response:", response.json())
  else:
      print(f"Failed with status code {response.status_code}: {response.text}")

# Replace with the Gemini API URL
api_url = "http://localhost:5000/generateChatTester"

# Replace with your API key if authentication is required
headers = {
    "Content-Type": "application/json",
}

prompt="// Focal class\npublic class Query {\nprotected String serverURL,\n    associatesID,\n    token = \"DSB0XDDW1GQ3S\", //don't change A4J won't work without this. Used for tracking tool use.\n    searchType,\n    type,\n    page,\n    offer;\nprotected ArrayList searchValues;\na4jUtil jawsUtil = new a4jUtil();\npublic Query(){}\n// Focal method\npublic String queryGenerator(String searchType, String type, String page, String offer, ArrayList searchValues) {\n        //    log.debug(\"queryGenerator - in\");\n        StringBuffer buffer = new StringBuffer();\n        buffer.append(serverURL);\n        buffer.append(\"?\");\n        buffer.append(\"t=\");\n        buffer.append(associatesID);\n        buffer.append(\"&\");\n        buffer.append(\"dev-t=\");\n        buffer.append(token);\n        buffer.append(\"&\");\n        buffer.append(searchType);\n        buffer.append(\"=\");\n        buffer.append(generateMultipleSearchString(searchType, searchValues));\n        buffer.append(\"&\");\n        buffer.append(\"type=\");\n        buffer.append(type);\n        buffer.append(\"&\");\n        buffer.append(\"offerpage=\");\n        buffer.append(page);\n        buffer.append(\"&\");\n        buffer.append(\"offer=\");\n        buffer.append(offer);\n        buffer.append(\"&\");\n        buffer.append(\"f=xml\");\n        //      log.debug(\"queryGenerator - out\");\n        return new String(buffer);\n    }\n}\nPlease infer the intention of the \"queryGenerator(String, String, String, String, ArrayList)\"."
prompt2sys="You are a senior tester in Java projects, your task is writting tests for a specific focal method in a focal class with JUnit5 and Mockito framework (A focal method means a method under test).\nI will provide the following information of the focal method:\n1. Required dependencies to import.\n2. The focal class signature.\n3. Source code of the focal method.\n4. Signatures of other methods and fields in the class.\nI will provide following brief information if the focal method has dependencies:\n1. Signatures of dependent classes.\n2. Signatures of dependent methods and fields in the dependent classes.\nYou need to create a complete unit test using JUnit 5, ensuring to cover all branches. Compile without errors, and use reflection to invoke private methods or fields if needed. No additional explanations required."
prompt2="// Focal class\npublic class Query {\nprotected String serverURL,\n    associatesID,\n    token = \"DSB0XDDW1GQ3S\", //don't change A4J won't work without this. Used for tracking tool use.\n    searchType,\n    type,\n    page,\n    offer;\nprotected ArrayList searchValues;\na4jUtil jawsUtil = new a4jUtil();\npublic Query(){}\n// Focal method\npublic String queryGenerator(String searchType, String type, String page, String offer, ArrayList searchValues) {\n        //    log.debug(\"queryGenerator - in\");\n        StringBuffer buffer = new StringBuffer();\n        buffer.append(serverURL);\n        buffer.append(\"?\");\n        buffer.append(\"t=\");\n        buffer.append(associatesID);\n        buffer.append(\"&\");\n        buffer.append(\"dev-t=\");\n        buffer.append(token);\n        buffer.append(\"&\");\n        buffer.append(searchType);\n        buffer.append(\"=\");\n        buffer.append(generateMultipleSearchString(searchType, searchValues));\n        buffer.append(\"&\");\n        buffer.append(\"type=\");\n        buffer.append(type);\n        buffer.append(\"&\");\n        buffer.append(\"offerpage=\");\n        buffer.append(page);\n        buffer.append(\"&\");\n        buffer.append(\"offer=\");\n        buffer.append(offer);\n        buffer.append(\"&\");\n        buffer.append(\"f=xml\");\n        //      log.debug(\"queryGenerator - out\");\n        return new String(buffer);\n    }\n}\nPlease infer the intention of the \"queryGenerator(String, String, String, String, ArrayList)\".\n"
prompt3=""
# Payload for the request
payload = {
"frequency_penalty": 0,
  "max_tokens": 1024,
  "presence_penalty": 0,
  "temperature": 0.5,
  "local_model":"01-ai/Yi-Coder-1.5B",
  "messages": [
        {
        "role": "system",
          "content": ""
        },
        {
        "role": "user",
          "content": prompt
        }
  ],
  "maxTokens": 1024
}

print(payload["local_model"])
# Send the POST request
response = requests.post(api_url, headers=headers, json=payload)

check_response(response)
#payload["local_model"] = "infly/OpenCoder-1.5B-Instruct"
#print(payload["local_model"])
#response = requests.post(api_url, headers=headers, json=payload)
#check_response(response)
#payload["local_model"] = "OpenCoder-8B-Instruct-Q6_K.gguf"
print(payload["local_model"])
#response = requests.post(api_url, headers=headers, json=payload)
#check_response(response)


