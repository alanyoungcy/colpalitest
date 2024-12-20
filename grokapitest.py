import asyncio

from xai_sdk.v1 import Client

from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

async def main():
    #"""Runs the example."""
    client = Client( api_key=XAI_API_KEY)

    prompt = "The answer to live and the universe is"
    print(prompt, end="")

    
    async for token in client.sampler.sample(max_len=3, prompt=prompt):
        print(token.token_str, end="")
    print("")


asyncio.run(main())

