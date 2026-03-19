"""Test reasoning settings for all 9 models via OpenRouter."""
import asyncio
import openai
from subtext.config import OPENROUTER_API_KEY

MODELS = {
    # Gemini
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite-preview",
    # Claude
    "opus-4.6": "anthropic/claude-4.6-opus-20260205",
    "sonnet-4.6": "anthropic/claude-4.6-sonnet-20260217",
    "haiku-4.5": "anthropic/claude-4.5-haiku-20251001",
    # GPT
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
}

client = openai.AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

async def test_model(key: str, model_id: str, effort: str) -> dict:
    """Test a single model with given reasoning effort."""
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=50,
            temperature=0.0,
            extra_body={"reasoning": {"effort": effort}},
        )
        content = response.choices[0].message.content
        # Check if reasoning tokens were used (via usage stats if available)
        usage = response.usage
        reasoning_tokens = getattr(usage, 'reasoning_tokens', None) or getattr(usage, 'completion_tokens_details', {})
        return {
            "key": key,
            "model_id": model_id,
            "effort": effort,
            "status": "success",
            "response": content[:100] if content else "",
            "usage": str(usage),
        }
    except Exception as e:
        return {
            "key": key,
            "model_id": model_id,
            "effort": effort,
            "status": "error",
            "error": str(e)[:200],
        }

async def main():
    print("Testing reasoning settings for all 9 models...\n")

    # Test with effort=none first
    results = []
    for key, model_id in MODELS.items():
        print(f"Testing {key} with effort=none...")
        result = await test_model(key, model_id, "none")
        results.append(result)

        if result["status"] == "error":
            print(f"  ERROR: {result['error']}")
            # Try minimal as fallback
            print(f"  Trying effort=minimal as fallback...")
            fallback = await test_model(key, model_id, "minimal")
            fallback["note"] = "fallback from none"
            results.append(fallback)
            if fallback["status"] == "success":
                print(f"  SUCCESS with minimal: {fallback['response']}")
            else:
                print(f"  FALLBACK ERROR: {fallback['error']}")
        else:
            print(f"  SUCCESS: {result['response']}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Group by model, show what works
    for key in MODELS:
        model_results = [r for r in results if r["key"] == key]
        none_result = next((r for r in model_results if r["effort"] == "none"), None)

        if none_result and none_result["status"] == "success":
            print(f"{key}: effort=none WORKS")
        else:
            minimal_result = next((r for r in model_results if r["effort"] == "minimal"), None)
            if minimal_result and minimal_result["status"] == "success":
                print(f"{key}: effort=minimal WORKS (none failed)")
            else:
                print(f"{key}: BOTH FAILED")

if __name__ == "__main__":
    asyncio.run(main())
