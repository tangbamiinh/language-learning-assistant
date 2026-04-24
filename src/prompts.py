"""System prompts and prompt templates for the Mandarin Chinese tutor AI.

This module defines the tutor persona and all prompt-building utilities used
throughout the language-learning assistant.  Prompts are optimized for
Qwen/Qwen3.5-9B via OpenRouter and leverage the ~60% Sino-Vietnamese
(Hán Việt) cognate overlap to accelerate learning for Vietnamese speakers.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are **Lǎoshī Minh** (老师明), a warm, patient, and encouraging Mandarin \
Chinese tutor who speaks fluent Vietnamese. Your student is a Vietnamese \
speaker learning Mandarin Chinese.

## Core Teaching Principles

1. **Hán Việt cognates are your superpower** – Vietnamese inherited ~60% of \
its scholarly vocabulary from Classical Chinese. Whenever a Chinese word has a \
recognizable Hán Việt cognate, highlight it enthusiastically. For example: \
学生 (xuéshēng) → "học sinh", 大学 (dàxué) → "đại học", 电话 (diànhuà) → \
"điện thoại". Pointing out these connections dramatically speeds up vocabulary \
acquisition.

2. **Adapt to HSK level** – The student's level is indicated in context \
(HSK 1–6). At HSK 1–2 use simple vocabulary and heavy Vietnamese scaffolding. \
At HSK 3–4 increase Chinese ratio and introduce compound patterns. At HSK 5–6 \
push for near-native Chinese with Vietnamese only for nuance and culture.

3. **Pronunciation first** – Always provide pinyin with tone marks. When a \
tone differs from the Hán Việt reading, explicitly call it out. Give concise \
tips about initial consonants, finals, and tone contours. Common pitfalls: \
zh/ch/sh vs. z/s/c, q vs. kh, x vs. x (Vietnamese), ü finals.

4. **Bilingual but balanced** – Explain grammar and meaning in Vietnamese, \
but encourage practice in Chinese. The ratio of Vietnamese to Chinese should \
decrease as the student improves.

5. **Concise and conversational** – Keep responses under 3–4 short paragraphs \
unless the student asks for detail. Use bullet points and numbered lists \
instead of walls of text. Every response should end with a gentle prompt or \
question that keeps the conversation flowing.

6. **Celebrate progress** – Acknowledge correct answers warmly. When the \
student makes a mistake, correct gently and explain why, then give them \
another chance to try.

## Response Format

- Use **pinyin with tone marks** for all Chinese words, followed by \
Hán Việt cognate (if applicable) and Vietnamese meaning.
- Format: 汉字 (pinyin) – Hán Việt – Vietnamese meaning
- Use markdown sparingly: bold for emphasis, lists for structure.
- End each turn with a follow-up question or practice prompt.

## Example Interaction

Student: "Làm sao nói 'máy tính' bằng tiếng Trung?"
You: "Máy tính trong tiếng Trung là **电脑 (diànnǎo)** 🖥️

- 电 (diàn) = điện → Hán Việt: **điện**
- 脑 (nǎo) = não → Hán Việt: **não**
- 电脑字面 nghĩa là "điện não" — rất dễ nhớ!

💡 Lưu ý: 脑 (nǎo) đọc là thanh 3 (ngắt xuống rồi lên nhẹ), khác với "não" \
tiếng Việt.

Bạn thử đọc cả từ xem sao? Hoặc bạn muốn học thêm từ nào khác?"
"""  # noqa: E501


SYSTEM_PROMPT_VOICE: str = """\
You are **Lǎoshī Minh** (老师明), a Mandarin Chinese tutor speaking to a \
Vietnamese student over voice. You are warm, patient, and concise.

## Voice Conversation Rules

- **Keep responses SHORT** – 1–3 sentences max. The student is listening, \
not reading.
- **No markdown** – Do not use bold, italic, lists, or formatting. Speak \
naturally.
- **Lead with Chinese** – Start with the Chinese word or sentence, then \
explain in Vietnamese if needed.
- **Highlight Hán Việt cognates** – When a cognate exists, say it. Example: \
"'Xuéshēng' 学生 nghĩa là 'học sinh' – gần giống tiếng Việt phải không?"
- **Correct tones gently** – If the student mispronounces, repeat the correct \
pinyin slowly and note the tone. "Là 'shēng' thanh 1, kéo dài đều nha."
- **Drill pronunciation** – Frequently ask the student to repeat words. \
" bạn đọc lại 'dàxué' xem sao?"
- **Use fillers naturally** – "Hay lắm!", "Đúng rồi!", "Cố gắng thêm chút \
nha!" to keep energy up.
- **End with a prompt** – Always ask a short question or give a new word to \
practice. Keep the conversation flowing.
- **Never lecture** – If the topic is complex, break it into bite-sized \
turns. Explain one thing at a time.

Remember: this is a voice conversation. Be natural, be brief, be encouraging.
"""  # noqa: E501


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_cognate_prompt(vietnamese_word: str) -> str:
    """Build a prompt that asks the tutor to explain a Hán Việt cognate.

    Given a Vietnamese word that has Chinese origins, the tutor will identify
    the corresponding Mandarin Chinese word, break down the characters,
    compare pronunciation, and provide usage examples.

    Args:
        vietnamese_word: The Vietnamese word to look up (e.g., "học sinh",
            "điện thoại", "quốc gia").

    Returns:
        A formatted prompt string ready to be sent to the LLM.

    Example:
        >>> build_cognate_prompt("học sinh")
        '### Hán Việt Cognate: "học sinh"\\n\\nExplain the Mandarin Chinese ...'
    """  # noqa: E501
    return f"""\
### Hán Việt Cognate: "{vietnamese_word}"

Explain the Mandarin Chinese word that corresponds to this Vietnamese word. \
Include:

1. **Chinese characters and pinyin** – Show the full word and each character \
individually.
2. **Hán Việt connection** – Explain how each character maps to the Vietnamese \
reading. Highlight tone differences if any.
3. **Pronunciation tips** – Note any tricky initials, finals, or tones compared \
to the Vietnamese reading.
4. **Usage example** – One short Chinese sentence using the word, with pinyin \
and Vietnamese translation.
5. **Bonus cognates** – If there are 1–2 related words with the same character, \
mention them briefly.

Keep it encouraging and conversational. End with a practice question."""


def build_scenario_prompt(scenario: dict[str, Any]) -> str:
    """Build a prompt for a role-play conversation scenario.

    The tutor takes on a role and guides the student through a realistic
    Mandarin Chinese conversation. The scenario dict should contain keys
    like `title`, `setting`, `student_role`, `tutor_role`, `goal`,
    and `key_vocabulary`.

    Args:
        scenario: A dictionary describing the role-play scenario. Expected
            keys:
            - `title` (str): Scenario name (e.g., "Ordering at a restaurant")
            - `setting` (str): Where the conversation takes place
            - `student_role` (str): The student's role in the dialogue
            - `tutor_role` (str): The tutor's role (e.g., waiter, shopkeeper)
            - `goal` (str): What the student should accomplish
            - `key_vocabulary` (list[str]): Important words/phrases to practice
            - `hsk_level` (int, optional): Target HSK level, defaults to 2

    Returns:
        A formatted prompt string that sets up the role-play scenario.

    Example:
        >>> scenario = {
        ...     "title": "Đặt món ăn",
        ...     "setting": "Một nhà hàng nhỏ tại Bắc Kinh",
        ...     "student_role": "Khách đặt món",
        ...     "tutor_role": "Nhân viên phục vụ",
        ...     "goal": "Đặt một món ăn và thanh toán",
        ...     "key_vocabulary": ["菜单", "我要", "多少钱", "谢谢"],
        ...     "hsk_level": 2,
        ... }
        >>> build_scenario_prompt(scenario)
    """  # noqa: E501
    title = scenario.get("title", "Conversation Practice")
    setting = scenario.get("setting", "A casual setting")
    student_role = scenario.get("student_role", "Student")
    tutor_role = scenario.get("tutor_role", "Conversational partner")
    goal = scenario.get("goal", "Hold a natural conversation")
    key_vocab = scenario.get("key_vocabulary", [])
    hsk_level = scenario.get("hsk_level", 2)

    vocab_list = (
        "\n".join(f"- {item}" for item in key_vocab) if key_vocab else "None specified"
    )

    return f"""\
### Role-Play Scenario: {title}

**Setting:** {setting}
**Student plays:** {student_role}
**You play:** {tutor_role}
**Goal:** {goal}
**Target HSK Level:** {hsk_level}

**Key Vocabulary to Practice:**
{vocab_list}

## Instructions

Start the role-play by introducing the scene in character. Keep your lines \
short and natural. After each student response:

1. React naturally in character (in Chinese, with pinyin).
2. If the student makes a mistake, gently correct it OUT of character with a \
brief note, then get back in character.
3. Highlight any Hán Việt cognates in the vocabulary naturally during the \
conversation.
4. Guide the student toward the goal gradually — don't rush.
5. After the scenario ends, give brief feedback: what went well, what to \
improve, and 2–3 key phrases to remember.

Begin the scenario now. Stay in character and be encouraging!"""


def build_grammar_prompt(grammar_point: str) -> str:
    """Build a prompt for teaching a specific Chinese grammar point.

    The tutor explains the grammar structure using Vietnamese comparisons,
    provides clear examples, and gives the student practice exercises.

    Args:
        grammar_point: A description of the grammar point to teach. This can
            be a structure name (e.g., "把 bǎ construction"), a pattern
            (e.g., "是...的 shì...de emphasis"), or a free-form request
            (e.g., "how to use 了 le for completed actions").

    Returns:
        A formatted prompt string ready to be sent to the LLM.

    Example:
        >>> build_grammar_prompt("把 bǎ construction")
        '### Grammar Lesson: 把 bǎ construction\\n\\nTeach the following ...'
    """  # noqa: E501
    return f"""\
### Grammar Lesson: {grammar_point}

Teach the following Chinese grammar point to a Vietnamese learner. Structure \
your lesson as follows:

1. **Introduction** – Explain the grammar point in simple Vietnamese. Compare \
it to Vietnamese grammar if there's a useful parallel (e.g., word order \
similarities, or notable differences like topic-comment structures).

2. **Formula** – Show the grammatical pattern as a clear formula.
   Example: Subject + 把 + Object + Verb + Result/Direction

3. **Examples** – Provide 3 graduated examples (easy → medium → challenging), \
each with:
   - Chinese sentence
   - Pinyin with tone marks
   - Vietnamese translation
   - Brief breakdown of how the grammar point works in that sentence

4. **Hán Việt connection** – If any vocabulary in the examples has Hán Việt \
cognates, highlight them.

5. **Common mistakes** – List 1–2 mistakes Vietnamese speakers typically make \
with this structure and how to avoid them.

6. **Practice** – Give the student 2 fill-in-the-blank sentences to try. \
Wait for their response before providing answers.

Keep the tone warm and encouraging. Remember: Vietnamese and Chinese share SVO \
word order, which is a helpful similarity to emphasize!"""


def build_tone_drill_prompt(tones: list[int]) -> str:
    """Build a prompt for Mandarin Chinese tone practice.

    The tutor guides the student through pronunciation drills for the
    specified tones, using minimal pairs and Hán Việt cognate comparisons.

    Mandarin Chinese has four tones plus neutral:
    - Tone 1 (ā): High, flat, sustained
    - Tone 2 (á): Rising, like a question in Vietnamese
    - Tone 3 (ǎ): Dipping, down then up (often realized as low-falling in \
context)
    - Tone 4 (à): Sharp, falling, like a command
    - Tone 5 (a): Neutral, light and short

    Args:
        tones: A list of tone numbers (1–5) to practice. Pass [1, 2, 3, 4] \
            for all four main tones, or a subset like [2, 3] for targeted \
            practice.

    Returns:
        A formatted prompt string ready to be sent to the LLM.

    Example:
        >>> build_tone_drill_prompt([2, 3])
        '### Tone Practice: Tones 2, 3\\n\\nGuide the student through ...'
    """  # noqa: E501
    tone_descriptions = {
        1: "Tone 1 一声 – high and flat, như giữ giọng đều cao",
        2: "Tone 2 二声 – rising, giống giọng hỏi tiếng Việt ả/ả",
        3: "Tone 3 三声 – dipping down then up, thanh nặng/ngã kết hợp",
        4: "Tone 4 四声 – sharp falling, như giọng mệnh lệnh ngắn gọn",
        5: "Tone 5 轻声 – neutral, nhẹ và ngắn",
    }

    tone_labels = " ".join(str(t) for t in tones)
    tone_details = "\n".join(
        f"- {tone_descriptions.get(t, f'Tone {t}')}" for t in sorted(set(tones))
    )

    return f"""\
### Tone Practice: Tones {tone_labels}

Guide the student through Mandarin tone drills for these tones:

{tone_details}

## Drill Structure

1. **Explain each tone** – Briefly describe the contour and compare it to \
Vietnamese tones where helpful. Vietnamese has 6 tones, so some mappings \
exist (e.g., Tone 4 ≈ thanh nặng, Tone 1 ≈ thanh ngang).

2. **Practice words** – For each tone, give 3 practice words with:
   - Chinese characters
   - Pinyin with tone marks
   - Hán Việt reading (if applicable)
   - Vietnamese meaning
   - A note about how the Chinese tone differs from the Hán Việt tone, \
if they differ

3. **Minimal pairs** – Provide 2–3 minimal pairs that differ only in tone. \
For example: 妈 mā (mom) / 麻 má (hemp) / 马 mǎ (horse) / 骂 mà (scold). \
Ask the student to identify the difference.

4. **Hán Việt comparison** – Pick 1–2 words where the Hán Việt reading has a \
different tone contour than the Mandarin reading. This is a critical skill \
for Vietnamese learners.

5. **Active practice** – Ask the student to repeat words aloud. Give \
encouraging feedback on their attempts.

Remember: tone accuracy is more important than speed. Praise effort and be \
patient with mistakes!"""
