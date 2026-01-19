- UK patterns (terms/assets/uk_patterns.json): 
    - based on the examples in examples/text/uk_telegram.out.txt, which patterns are matching low-value / non-context-specific content that you would recommend removing?
- Scoring: Please review score.py and the example output in examples/text/uk_telegram.out.txt, and suggest changes that would make substantively meaningful terms rank higher—specifically, war-related terms in this case—rather than the generic phrases that appear in almost any context.




    - VERB patterns (VERB-NOUN, VERB-ADJ-NOUN, etc.) may be too broad
        - captures action phrases like "збирати АК", "займаються вишколом" instead of noun phrases
        - consider restricting to participle forms only using MORPH constraints