# kimi_api_key = 'sk-at16DOyUcZ025r6cbh30uHUoehiwfII50W73hnVnGdPxgjsc'
# kimi_api_key = 'sk-nDRrC3NSzRUTJZV9Mk3I9W3mVMrYiL8SupqQWJHy83a7XL0M'

from openai import OpenAI
 
client = OpenAI(
    api_key = "sk-",
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "kimi-k2-0711-preview",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.6,
)
 
print(completion.choices[0].message.content)