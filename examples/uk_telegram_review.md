# Ukrainian Telegram Post Extraction Review

## Test File
`examples/texts/uk_telegram.txt` - a post about infantry quality degradation in prolonged wars.

## Summary
**Extracted: 24 phrases** (14 unique maximal)
**Quality: Mixed** - captures key phrases from subject positions but scope is limited.

---

## What the Tool Gets Right

### Core noun phrase patterns work well
- `Кількість кадрових солдатів` (NOUN-ADJ-NOUN)
- `Підготовка особового складу` (NOUN-ADJ-NOUN)
- `початку війни` (NOUN-NOUN)
- `місяць вишколу` (NOUN-NOUN)
- `гарний мікроклімат` (ADJ-NOUN)
- `швидкій адаптації новачків` (ADJ-NOUN-NOUN)
- `розбудові бойового побратимства` (NOUN-ADJ-NOUN)
- `Радянських рудиментів в ЗСУ` (ADJ-NOUN-ADP-NOUN)
- `український вояк` (ADJ-NOUN)
- `3-тя ОШБ` (ADJ-NOUN) - ordinal + acronym handled correctly

### Redundancy resolution works
- Family grouping correct: sub-phrases properly nested under maximal spans

---

## Issues Found

### 1. Fixed: PART-NOUN pattern removed
Was capturing meaningless phrases like `тільки місяць` ("only a month").
**Action taken**: Removed PART-NOUN and PART-ADJ-NOUN patterns from `uk_patterns.json`.

### 2. Documented: VERB patterns too broad
See `additional_tasks.md` - patterns like VERB-NOUN capture action phrases instead of noun phrases.

### 3. Documented: Subject subtree scope limitation
See `additional_tasks.md` - extraction is limited to subject subtrees regardless of `exclusive_search` setting.

---

## Phrases Correctly Extracted

All extracted phrases are legitimate key noun phrases from subject positions in the text. The patterns in `uk_patterns.json` work correctly for their intended scope.

## Phrases Not Extracted (due to scope, not pattern issues)

These exist in object/prepositional positions, outside subject subtrees:
- затяжних воєн, бойових втрат, мотивованих резервістів
- повномасштабного вторгнення, важких ушкоджень
- сучасному піхотному бою, закордонних полігонах
- бойового злагодження, бойового духу, якість піхоти
