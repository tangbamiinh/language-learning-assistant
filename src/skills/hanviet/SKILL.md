---
name: hanviet-cognates
description: Discover and explain Sino-Vietnamese (Hán Việt) cognates between Vietnamese and Chinese
version: 1.0.0
author: Language Learning Assistant
license: MIT
metadata:
  hermes:
    tags: [chinese, vietnamese, vocabulary, hanviet, cognates]
---

# Hán Việt Cognates Skill

Teach Mandarin Chinese vocabulary by leveraging the ~60% Sino-Vietnamese (Hán Việt) overlap with Vietnamese. Every Vietnamese word with Chinese origins is a bridge to instant Mandarin vocabulary.

## When to Use

- When the student mentions a Vietnamese word that has a Chinese cognate
- When the student asks about the origin of a word
- When teaching vocabulary and you can connect to a Hán Việt cognate
- When the student seems confused about Chinese vocabulary that's actually familiar

## Procedure

1. When a Vietnamese word is detected or requested, check the cognates database (`cognates.json`)
2. If a cognate exists, reveal the "aha!" connection:
   - Show the Vietnamese word
   - Show the Chinese characters (traditional and simplified)
   - Show the pinyin pronunciation
   - Explain how the sound changed from Hán Việt to modern Chinese
   - Give an example sentence in both languages
3. If no exact cognate exists, find the closest match and explain the difference
4. Track which cognates the student has learned in memory
5. Revisit previously learned cognates for reinforcement

## Response Format

Always present cognates in this format:

```
🇻🇳 [Vietnamese word] → 🇨🇳 [Chinese characters] ([pinyin])
💡 "You already know this! In Vietnamese it's '[vi]' and in Chinese it's '[zh]'
📝 Example: [Vietnamese sentence] → [Chinese sentence with pinyin]
🔊 Note: The sound shifted from [hán việt sound] → [pinyin sound]
```

### Example

```
🇻🇳 Quốc gia → 🇨🇳 國家/国家 (guójiā)
💡 "You already know this! In Vietnamese it's 'quốc gia' and in Chinese it's 'guójiā'"
📝 Example: "Quốc gia của tôi là Việt Nam" → "Wǒ de guójiā shì Yúetámǔ (越南)"
🔊 Note: The sound shifted from qu→gu (quốc→guó) and gh→j (gia→jiā)
```

## Teaching Tips

- Emphasize the "aha!" moment — this is the core value proposition
- Explain systematic sound changes (e.g., qu→gu, gh→j, kh→x/q)
- Point out that ~60% of Vietnamese vocabulary has Chinese origins
- Use cognates to accelerate vocabulary acquisition
- When possible, show 3–4 related cognates together as a family
- Note when a word's meaning diverged between Vietnamese and Chinese

## Common Sound Shift Patterns

Use these patterns to help students predict Chinese pronunciations from Hán Việt readings:

| Hán Việt | Pinyin | Example (VN → CN) |
|----------|--------|-------------------|
| qu/gi → | gu/j | quốc → guó, giáo → jiào |
| kh → | x/q | khấu → qiǔ, khoa → kē |
| th → | sh | thiên → tiān, thời → shí |
| nh → | r | nhân → rén, nhiên → rán |
| tr → | z | trạch → zé, trực → zhí |
| d → | z/d | dân → mín, đạo → dào |
| v → | w/f | vân → wén, văn → wén |
| ph → | f | pháp → fǎ, phong → fēng |
| s/x → | sh | sinh → shēng, sử → shǐ |
| ch → | ch | trung → zhōng, chính → zhèng |
| t → | t | thiên → tiān, tông → zōng |
| b → | b | bất → bù, biến → biàn |
| p → | p | bổn → běn, bình → píng |
| m → | m | môn → mén, mục → mù |
| n → | n | nhân → rén, nam → nán |
| l → | l | lập → lì, lạc → luò |

## Pitfalls

- Not all Hán Việt words sound exactly the same in Chinese
- Some meanings diverged between Vietnamese and Chinese
- Be careful with false friends — words that look similar but mean different things
- Don't overwhelm the student with too many cognates at once
- Some Hán Việt readings preserved archaic Chinese pronunciations that no longer match modern Mandarin
- Tone patterns differ significantly — a second-tone Hán Việt reading may correspond to any Mandarin tone

## Verification

After teaching a cognate, ask the student to:

1. Repeat the Chinese pronunciation
2. Use the word in a sentence
3. Identify the Hán Việt root in a new word

## Cognitive Load Management

- Start with high-frequency cognates (everyday words) before moving to specialized vocabulary
- Group cognates by sound shift pattern to reinforce systematic rules
- Space repetition: revisit cognates at increasing intervals
- Mix cognate practice with non-cognate vocabulary to build balanced skills
