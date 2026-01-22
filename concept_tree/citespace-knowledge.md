# CiteSpace noun phrases and concept trees: extracted details

This note consolidates everything found in the local PDFs/text about how CiteSpace extracts noun phrases and concept trees. It sticks to what is explicitly stated in the sources (no inference beyond that), and cites the file locations where each detail appears.

## Noun phrases: extraction pipeline (CiteSpace UI + algorithmic details)

### Inputs / sources

- Titles and abstracts are the primary text fields for noun phrases in WoS-like records; PubMed/NBIB uses TI and AB fields. (`txt/howtousecitespace.txt`, `txt/ch16 PubMed.txt`)
- CiteSpace can also extract noun phrases from citation contexts (described in several methodological papers). (`txt/Chen et al - 2010 - The Structure and Dynamics of Cocitation Clusters.txt`, `txt/2009.08374.txt`, `txt/frma-05-607286.txt`)
- Keywords are a distinct source and are not “noun phrases” by default. (`txt/howtousecitespace.txt`)

### Configuration and UI steps

- In CiteSpace, choose **Term Type: Noun Phrases** and **Node Type: Term** to build a network of co-occurring terms. (`txt/howtousecitespace.txt`)
- CiteSpace will prompt to create POS tags for the project. If tags exist, you can reuse or re-run them; re-run is recommended after upgrades. (`txt/howtousecitespace.txt`)
- After choosing Create POS Tags, CiteSpace preprocesses data and reports valid years/records in the Space Status window. (`txt/howtousecitespace.txt`)
- Press **GO!** to start noun-phrase extraction. The process is slow and progress is reported in the Space Status panel. (`txt/howtousecitespace.txt`)
- Once extraction finishes, CiteSpace can visualize the co-occurrence network of terms (noun phrases). (`txt/howtousecitespace.txt`)

### POS tagging (what, how, with what tools)
Note - I STILL don't know what he uses, but I think spaCy is perfectly fine for PoS-tagging, the real 'mystery' lies in what he does with it THEN


- POS tagging annotates every word with a POS tag (example: `tree/nn`, `grow/vb`) and is the input to pattern matching. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- POS tagging is explicitly called out as time-consuming. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- NLP tools for POS tagging are referenced; GATE is listed as an example tool. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- In the SDSS study, POS tagging is explicitly done using Stanford NLP taggers (Toutanova & Manning) for titles/abstracts. (`txt/Zhang et al. - 2011 - Scientometrics of Big Science A Case Study of Res.txt`)
- In PaperPoles (related work), POS tagging (Toutanova & Manning) is used to identify nouns prior to phrase extraction. (`txt/He et al. - 2019 - PaperPoles Facilitating adaptive visual explorati.txt`)

### Heuristics and extraction logic

- Noun phrase extraction follows heuristics (simple to complex
So THIS is what we have to find out more about (and I'm drafting him an email, asking one more time (last time was in 2022...)
Dear Chaomei,
We’re once again taking a close look at everything you’ve written about your 'magical' noun phrase approach because we want to replicate it for Russian and Ukrainian (and we’d be more than happy to share results with you once we reach comparable quality). Your 2007 Turning Points remains the most useful source we can find, but the exact regular expressions used for the subject–predicate pattern (the 3,480‑character regex shown in Fig. 7.12) are only visible as an image in the book and not explained in detail in the text, so we cannot reproduce or implement them precisely.

Would you be willing to share the full regular expression(s) (or the underlying code/snippet) used for the noun-phrase extraction patterns in CiteSpace? Even a text dump of the regex would make a big difference for us. We’re trying to replicate the exact behavior to compare against our own pipelines and to understand why our noun-phrase quality is not yet close to CiteSpace’s.

Also, if there are additional heuristics beyond the POS/regex patterns (e.g., filters, lexicons, or any post-processing steps) that affect noun-phrase quality, we would be very grateful for any pointers. I assume the “magic” is not just in the subject–predicate pattern itself, so any guidance on other key steps would help us a lot.

Thanks again for your guidance in all of this.
 
Kindest regards,
-Stephan
P.S. I do keep showcasing CiteSpace in all of the workshops we give around the Transatlantic space on our RuBase/StratBase approach, and I hope that will send more customers your way! 

) over POS-tagged text. Output is a list of noun phrases and their frequencies. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- Noun phrases are treated as more meaningful, self-contained lexical units than single words. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- A noun phrase ends with a head noun; modifiers can be adjectives or nouns. Examples include adjective+noun, noun+noun chains, and multiword noun phrases (e.g., “word-word-word-noun” / “word-word-noun-noun”). (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`, `txt/Zhang et al. - 2011 - Scientometrics of Big Science A Case Study of Res.txt`)
- Users can filter noun phrases by the number of nouns in a phrase. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)

### Length constraints

- Default noun-phrase length in CiteSpace is **2–4 words**; single words or 5+ word phrases are excluded unless settings are changed. (`txt/howtousecitespace.txt`, `txt/ch16 PubMed.txt`)
- The min/max word counts are configurable in project properties before extraction. (`txt/howtousecitespace.txt`, `txt/ch16 PubMed.txt`)

### Language constraints

- CiteSpace extracts noun phrases from English text but not from Chinese text (CSSCI/CNKI). (`txt/ch11 CSSCI.txt`, `txt/howtousecitespace-6.1.R2 in progress 1-4, 11-12, 16.txt`)

## Noun phrases: pattern matching and regex building blocks (detailed)
THIS is what we have to really get to the bottom of


Source for explicit pattern definitions: `txt/Chen - 2011 - Turning Points The Nature of Creativity.txt` (Table 7.6 and Fig. 7.12).

Pattern matching is done via **regular expressions over POS-tagged text**. The subject-predicate regex is stated to be **3,480 characters long**. Only the building blocks are explicitly enumerated in text; the full regex is shown as a figure image.

### Explicit pattern building blocks (Table 7.6)

- **noun**: `(article OR adjective)* + word/nn[sp]*`  
  Example: “heavy and cold rain”  
  Length: 181 characters

- **noun phrase**: various combinations of nouns and other types of words  
  Example: “information visualization research”  
  Length: 858 characters

- **subject**: `noun OR noun phrase OR word/prp`  
  Example: “this article”  
  Length: 1,057 characters

- **simple verb**: `word/vb[dzpn[^g]]`  
  Example: “discover”  
  Length: 27 characters

- **verb**: various combinations of verbs  
  Example: “could have been discovered”  
  Length: 257 characters

- **concept**: noun phrase, including noun of noun  
  Example: “exotic plant”  
  Length: 858 characters

- **predicate**: `subject + verb + (noun phrase OR noun OR word/vbg)`  
  Example: “we + introduce + a new algorithm”  
  Length: 3,480 characters

### Concept tree construction heuristics (from the same section)

- Phrase decomposition uses head nouns: e.g., “large-scale network” is split into head noun “network” and modifier “large-scale”; head noun becomes parent, modifier becomes child. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- Phrases with the same head noun are aligned; their modifiers become children (e.g., “heterogeneous information space” and “exploration of an information space” share head noun “information space”). (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)
- Articles (a/an/the) are omitted in tree representations. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)

## Concept trees: extraction pipeline and mechanics

### Core recipe (explicit steps)

From `txt/frma-05-607286.txt`:

1) Extract noun phrases from POS-tagged text (titles/abstracts/citation contexts).  
2) Derive hierarchical relations at sentence level: if noun phrase nA co-occurs with nB and nC, nA is higher-level.  
3) Visualize the hierarchy as a concept tree.
We SHOULD be able to replicate this, no?


### Concept tree construction from clusters

From `txt/Chen and Song - 2017 - Representing Scientific Knowledge The Role of Unc.txt`:

- Terms are extracted from titles/abstracts/keywords of citing articles, then filtered to cluster-specific terms.
- Co-occurrence of filtered terms is used to build a hierarchical structure.
- Hierarchy can be derived using m-reachability

; higher reachability => higher position.
- The concept tree is treated as a proxy ontological representation of a cluster.

### Concept trees from citation contexts

- Nodes are noun phrases extracted from citation contexts of member references in a cluster; branches highlight themes. (`txt/frma-05-607286.txt`, `txt/2009.08374.txt`)
- Concept trees can be built for a cluster, a single reference, or a specific word/phrase. (`txt/frma-05-607286.txt`, `txt/2009.08374.txt`)

### Text sources that can feed concept trees

From `txt/howtousecitespace.txt` and `txt/Chen - 2019 - How to Use CiteSpace.txt`:

- Pasted text into an input window
- Full-text files
!

- Folder of WoS-format files (downloaded data or intermediate files after clustering)

### MySQL-based workflows (CNKI/PubMed)

- CNKI concept tree visualization requires MySQL on localhost; supported in Advanced version. (`txt/ch12 CNKI.txt`)
- PubMed concept trees: import NBIB data to MySQL, extract noun phrases, then build concept tree; progress is reported in command line. (`txt/ch16 PubMed.txt`)

### Interaction and visualization behavior

- Tree window + context window; hovering a node shows contexts with highlights. (`txt/howtousecitespace.txt`, `txt/ch12 CNKI.txt`, `txt/ch16 PubMed.txt`)
- General concepts near root, specific concepts farther along branches; bold nodes indicate saliency/frequency in CNKI example. (`txt/ch12 CNKI.txt`)
- Zoom/pan interactions described in manuals. (`txt/howtousecitespace.txt`, `txt/ch12 CNKI.txt`)

### Scalability notes

- Concept/predicate trees are implemented with prefuse DOITree; node-and-link layout chosen.
- Response rates decrease for concept trees with >200,000 nodes; pruning recommended. (`txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)

## Visual evidence captured from PDF pages 207-208

- Images exported from the 2011 book are stored at:
  - `txt/page_images/turning_points_p-207.png`
  - `txt/page_images/turning_points_p-208.png`
- Page 207 (Fig. 7.11) shows the workflow: select sources (Abstracts/Articles/Books) -> data selection -> POS tagging -> pattern matching -> noun phrases (concepts) + subject/predicates -> tree construction -> merge with existing tree -> visualization -> optional add another source. The figure also includes a notes bubble referencing: extend semantic cluster processing, structural starting point, experiment platform, basis for interactive controls, comprehensive representation, divide-and-conquer strategy, significant contribution, identifying key changes in visually salient features.
- Page 208 (Fig. 7.12 + Table 7.6) contains the subject-predicate regex illustration and the explicit pattern building blocks listed above; the regex itself is shown as an image only (not text).

## Other contextual facts tied to noun phrases

- Cluster labels: candidates are noun phrases and index terms from citing articles; ranked with tf*idf, LLR, mutual information. (`txt/Chen et al - 2010 - The Structure and Dynamics of Cocitation Clusters.txt`)
- In CiteSpace manuals, LLR often yields the best labels (uniqueness/coverage). (`txt/howtousecitespace.txt`)
- Noun phrases are used for burst detection and other term-level analyses. (`txt/Chen and Song - 2017 - Representing Scientific Knowledge The Role of Unc.txt`, `txt/Chen - 2011 - Turning Points The Nature of Creativity.txt`)

## Limits of evidence found

- The full 3,480-character regex for subject-predicate extraction is not available as text in the PDFs; it appears only as a figure image in the book. The extracted text includes only the building blocks in Table 7.6, not the full regex.
