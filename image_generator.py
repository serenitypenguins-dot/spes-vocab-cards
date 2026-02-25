import asyncio
import os
import uuid
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STYLE_DESCRIPTIONS = {
    "cartoon": "Bright, simple cartoon illustration, kid-friendly, white background, no text",
    "watercolor": "Soft watercolor painting style, white background, no text",
    "linedrawing": "Clean black line drawing, minimal detail, white background, no text",
    "realistic": "Realistic photograph style, clean white background, no text",
}

GRADE_LABELS = {
    "k": "Kindergarten",
    "1": "1st grade",
    "2": "2nd grade",
    "3": "3rd grade",
    "4": "4th grade",
    "5": "5th grade",
}

# Determine which provider to use
_openai_key = os.getenv("OPENAI_API_KEY")
_gemini_key = os.getenv("GEMINI_API_KEY")

# Prefer OpenAI (more reliable), fall back to Gemini
PROVIDER = "openai" if _openai_key else "gemini"
print(f"Image provider: {PROVIDER}")


def _build_prompt(word: str, grade: str, style: str) -> str:
    style_desc = STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["cartoon"])
    grade_label = GRADE_LABELS.get(grade, "Kindergarten")
    return (
        f"A {style_desc} of the concept '{word}', suitable for {grade_label} students. "
        f"Simple, clear, centered on white background, no text or letters in the image."
    )


async def _generate_openai(prompt: str, output_dir: Path) -> Path | None:
    """Generate image using OpenAI gpt-image-1."""
    import openai
    client = openai.AsyncOpenAI(api_key=_openai_key)
    result = await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="low",
        n=1,
    )
    img_data = base64.b64decode(result.data[0].b64_json)
    img_path = output_dir / f"{uuid.uuid4().hex}.png"
    img_path.write_bytes(img_data)
    return img_path


async def _generate_gemini(prompt: str, output_dir: Path) -> Path | None:
    """Generate image using Gemini 3 Pro Image."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_gemini_key)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3-pro-image-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(image_size="1K"),
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_path = output_dir / f"{uuid.uuid4().hex}.png"
            img_path.write_bytes(part.inline_data.data)
            return img_path
    return None


async def generate_image(word: str, grade: str, style: str, output_dir: Path, retries: int = 2) -> Path | None:
    prompt = _build_prompt(word, grade, style)
    gen_fn = _generate_openai if PROVIDER == "openai" else _generate_gemini
    for attempt in range(retries + 1):
        try:
            result = await gen_fn(prompt, output_dir)
            if result:
                return result
            if attempt < retries:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating image for '{word}' (attempt {attempt+1}): {e}")
            if attempt < retries:
                await asyncio.sleep(2)
    return None


async def generate_images_batch(
    words: list[str],
    grade: str,
    style: str,
    output_dir: Path,
    progress_callback=None,
    batch_size: int = 4,
) -> dict[str, Path | None]:
    """Generate images for all words. Returns {word: path_or_none}."""
    results: dict[str, Path | None] = {}
    total = len(words)

    for i, word in enumerate(words):
        # Update progress BEFORE generating (so UI shows "generating X...")
        if progress_callback:
            await progress_callback(i, total, word)

        path = await generate_image(word, grade, style, output_dir)
        results[word] = path

    # Final progress update
    if progress_callback:
        await progress_callback(total, total, words[-1])

    return results
