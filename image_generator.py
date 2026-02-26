import asyncio
import os
import re
import uuid
import base64
import json
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

_openai_key = os.getenv("OPENAI_API_KEY")
_gemini_key = os.getenv("GEMINI_API_KEY")

PROVIDER = "openai" if _openai_key else "gemini"
print(f"Image provider: {PROVIDER}")


def parse_word_input(raw: str) -> tuple[str, str | None]:
    """Parse 'strike (lightning strike)' → ('strike', 'lightning strike').
    Returns (word, context_or_none)."""
    match = re.match(r'^(.+?)\s*\((.+?)\)\s*$', raw.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return raw.strip(), None


def _build_prompt(word: str, grade: str, style: str, context: str | None = None) -> str:
    style_desc = STYLE_DESCRIPTIONS.get(style, STYLE_DESCRIPTIONS["cartoon"])
    grade_label = GRADE_LABELS.get(grade, "Kindergarten")
    if context:
        return (
            f"A {style_desc} illustrating the meaning of '{word}' as in '{context}', "
            f"suitable for {grade_label} students. "
            f"Simple, clear, centered on white background, no text or letters in the image."
        )
    return (
        f"A {style_desc} of the concept '{word}', suitable for {grade_label} students. "
        f"Simple, clear, centered on white background, no text or letters in the image."
    )


async def get_word_meanings(word: str, grade: str, num_meanings: int = 3) -> list[str]:
    """Use OpenAI to get multiple meanings of a word, suitable for the grade level."""
    import openai
    client = openai.AsyncOpenAI(api_key=_openai_key)
    grade_label = GRADE_LABELS.get(grade, "Kindergarten")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"List the {num_meanings} most common meanings of the word '{word}' "
                f"that a {grade_label} student might encounter. "
                f"For each meaning, write a short 2-5 word description that could be illustrated as a picture. "
                f"Return ONLY a JSON array of strings, nothing else. "
                f'Example for "bat": ["a flying animal bat", "a baseball bat for hitting", "to bat your eyelashes"]'
            )
        }],
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    # Parse JSON array from response
    try:
        # Handle markdown code blocks
        if "```" in text:
            text = re.search(r'\[.*?\]', text, re.DOTALL).group()
        meanings = json.loads(text)
        if isinstance(meanings, list) and all(isinstance(m, str) for m in meanings):
            return meanings[:num_meanings]
    except Exception:
        pass
    # Fallback: just return the word itself
    return [word]


async def _generate_openai(prompt: str, output_dir: Path) -> Path | None:
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


async def generate_image(word: str, grade: str, style: str, output_dir: Path,
                         context: str | None = None, retries: int = 2) -> Path | None:
    prompt = _build_prompt(word, grade, style, context)
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


# Card = (display_word, meaning_label_or_none, image_path)
Card = tuple[str, str | None, Path | None]


async def generate_cards(
    raw_words: list[str],
    grade: str,
    style: str,
    output_dir: Path,
    multi_meanings: bool = False,
    progress_callback=None,
) -> list[Card]:
    """Generate cards. Returns list of (word, meaning, image_path).
    If multi_meanings=True and no context given, generates multiple cards per word."""

    # First, parse all inputs
    parsed = [parse_word_input(w) for w in raw_words]

    # Build the full card list
    card_specs: list[tuple[str, str | None]] = []  # (word, context)
    for word, context in parsed:
        if context:
            # User gave specific meaning — just use it
            card_specs.append((word, context))
        elif multi_meanings:
            # Get multiple meanings from AI
            meanings = await get_word_meanings(word, grade)
            for meaning in meanings:
                card_specs.append((word, meaning))
        else:
            card_specs.append((word, None))

    total = len(card_specs)
    cards: list[Card] = []

    for i, (word, context) in enumerate(card_specs):
        if progress_callback:
            label = f"{word}" if not context else f"{word} ({context})"
            await progress_callback(i, total, label)

        path = await generate_image(word, grade, style, output_dir, context)
        cards.append((word, context, path))

    if progress_callback:
        await progress_callback(total, total, "done")

    return cards
