from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync, ExLlamaV2Sampler
from runpod.serverless import start
from config import get_config
import asyncio

cfg = get_config()

model = ExLlamaV2(cfg)
cache = ExLlamaV2Cache(lazy=True)
tokenizer = ExLlamaV2Tokenizer(cfg)

gen = ExLlamaV2DynamicGeneratorAsync(model, cache, tokenizer)

async def handler(event):
    text = event["input"]["prompt"]
    tokens = tokenizer.encode(text)

    job = ExLlamaV2DynamicJobAsync(
        generator=gen, 
        input_ids=tokens,
        gen_settings=ExLlamaV2Sampler.Settings()
    )
    
    async for result in job:
        text = tokenizer.decode(result.token_ids)
        yield {"text": text}
