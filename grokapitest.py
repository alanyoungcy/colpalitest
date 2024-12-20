import asyncio

from xai_sdk.v1 import Client

XAI_API_KEY = "xai-VslBZI8ZR8PtKR90LDO2hGo4nDKVuP9ckBvqeXe79v9aIj5EQvxntEO1D8PoTOwm6scChW7bpvRh6zyQ"

async def main():
    #"""Runs the example."""
    client = Client( api_key=XAI_API_KEY)

    prompt = "The answer to live and the universe is"
    print(prompt, end="")

    
    async for token in client.sampler.sample(max_len=3, prompt=prompt):
        print(token.token_str, end="")
    print("")


asyncio.run(main())

