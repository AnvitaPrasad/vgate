"""Frozen prompt templates. Changing any string here changes PROMPT_ID and invalidates cached runs.

Layout is always [image] -> [evidence, if any] -> [instruction + question] so the NE and EV calls share
the [system][image] prefix (prefix/encoder caching on the A2000 amortises the gate's extra NE pass).
"""
import hashlib

SHORT = "Answer the question using a single word or phrase."
NE = SHORT + "\nQuestion: {question}"
EV = ("Context:\n{context}\n\n" + SHORT + " Use the image and the context if it is relevant."
      "\nQuestion: {question}")
ENT = ("What is the specific name of the main entity shown in this image (e.g. the species, landmark, "
       "building, product or object)? Answer with the name only.")
PTRUE = ("Question: {question}\nProposed answer: {answer}\n"
         "Is the proposed answer correct? Answer with yes or no only.")
PARA = SHORT + "\nQuestion (rephrased): {question}"  # paraphrase-perturbation feature (question lower-cased)

PROMPT_ID = hashlib.sha256("\x00".join([NE, EV, ENT, PTRUE, PARA]).encode()).hexdigest()[:8]

# --- PREREG v3 (passage-level selection). Kept out of PROMPT_ID so that the id of the earlier runs does not change.
ONE_PASSAGE_WORDS = 300  # a single passage may be longer than one of five (E-VQA sections: median 108 words, p90 353)
PV = ("Context:\n{context}\n\nQuestion: {question}\nIs this context about what the image shows, and does it contain "
      "the answer to the question? Answer with yes or no only.")
EV_KEEP = ("Context:\n{context}\n\n" + SHORT + " Your answer without the context was: {answer}. Keep that answer unless "
           "the context is clearly about what the image shows and supports a different answer.\nQuestion: {question}")
PROMPT_ID_V3 = hashlib.sha256("\x00".join([EV, PV, EV_KEEP, str(ONE_PASSAGE_WORDS)]).encode()).hexdigest()[:8]


# --- PREREG v5 (pilot): a released passage critic, aimagelab/ReAG-Critic (Compagnoni et al., arXiv 2511.22715), scores one
# retrieved passage at a time. System and user texts are copied verbatim from its model card (read 2026-10-05).
CV_SYSTEM = """You are a multimodal reasoning assistant specialized in Knowledge-Based Visual Question Answering (KB-VQA).
Your task is to evaluate whether a given text passage provides useful and relevant information for answering a question about an image.

You will be given:
- Image: a visual scene containing entities, actions, and context.
- Question: a natural-language question that refers to the image.
- Text Passage: an external knowledge snippet retrieved from a database or the web.

You must analyze the semantic alignment between the text, the image, and the question.
Follow these steps carefully before giving your final decision:
1. Understand the visual scene: Identify the key objects, people, actions, and context visible in the image.
2. Interpret the question: Determine what information the question seeks.
3. Analyze the text passage: Extract the main claims, facts, and entities mentioned in the text.

Compare for relevance: Assess whether the information in the text:
- Contains at least one sentence that supports answering the question about the image, OR
- Provides background knowledge needed to interpret or reason about the image-question pair.

Important:
- If even a single sentence in the passage is relevant or useful, consider the entire passage as relevant and answer "Yes".
- If no part of the passage contributes meaningfully to answering the question, answer "No".

Output only one word:
"Yes" -> if the text provides relevant or useful information for answering the question.
"No" -> if the text is irrelevant or unhelpful."""
CV_USER = """Here is the question on the image above:
{question}

Here is the text passage to analyze:
{passage}

Does the text passage contain at least one sentence that may have some information useful to answer the user question?
"Yes"/"No" answer:"""
PROMPT_ID_V5 = hashlib.sha256("\x00".join([CV_SYSTEM, CV_USER, str(ONE_PASSAGE_WORDS)]).encode()).hexdigest()[:8]


# --- PREREG v7 (CVQA, four-option multiple choice). Kept out of PROMPT_ID; same layout as NE / EV above.
MC = "Answer with the letter of the correct option (A, B, C or D) only."
NE_MC = MC + "\nQuestion: {question}\n{options}"
EV_MC = ("Context:\n{context}\n\n" + MC + " Use the image and the context if it is relevant."
         "\nQuestion: {question}\n{options}")
PROMPT_ID_V7 = hashlib.sha256("\x00".join([NE_MC, EV_MC]).encode()).hexdigest()[:8]


def options_block(options):
    return "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options))


# --- PREREG v8: published retrieve / skip prompts run with our frozen readers (text copied from the sources, read 2026-10-06).
# gate_chen: the search-need prompt of Chen et al. (arXiv 2502.18023, Appendix A.1, hard boundary; question, then image).
# gate_urag: the training-free router prompt of UniversalRAG (arXiv 2504.20734; route/prompt.py of wgcyeo/UniversalRAG);
#            the published router reads the text query only and replies "no" when retrieval is not needed.
GATE_CHEN = 'You are an assistant capable of deciding whether a search is needed in a multimodal question-answering scenario. Below, I will provide you with a multimodal question that includes a text question and an image link.\nPlease respond with "true" or "false," indicating whether a search is necessary (true) or not (false) to answer this multimodal question.\nText question: {question}\nImage: '
GATE_URAG = 'Classify the following query into one or more categories from: [no, paragraph, document, table, image, clip, video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:\n- no: The query can be answered directly with common knowledge, reasoning, or computation without external data.\n- paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.\n- document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.\n- table: The query requires information that is best represented in a tabular format, often involving comparisons or structured data.\n- image: The query focuses on visual aspects like appearances, structures, or spatial relationships.\n- clip: The query targets a short, specific moment or event within a video, without needing full context.\n- video: The query requires understanding dynamic events, motion, or sequences over time in a video.\n\nFor cross-modality queries that require multiple types of information, combine categories with \'+\' (e.g., paragraph+image, document+video).\n\nExamples:\n1. "What is the capital of France?" -> no\n2. "What is the birth date of Alan Turing?" -> paragraph\n3. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" -> document\n4. "Among the recipients of the Turing Award, who had the earliest birth year?" -> table\n5. "Describe the appearance of a blue whale." -> image\n6. "Describe the moment Messi scored his goal in the 2022 World Cup final." -> clip\n7. "Explain how Messi scored his goal in the 2022 World Cup final." -> video\n8. "Solve 12 x 8." -> no\n9. "Who played a key role in the development of the iPhone?" -> paragraph\n10. "Which Harvard University graduate played a key role in the development of the iPhone?" -> document\n11. "What is the cheapest iPhone model available in 2023?" -> table\n12. "Describe the structure of the Eiffel Tower." -> image\n13. "Describe the moment Darth Vader reveals he is Luke\'s father in Star Wars." -> clip\n14. "Analyze the sequence of events leading to the fall of the Empire in Star Wars." -> video\n15. "Describe the visual appearance and habitat of the blue whale." -> paragraph+image\n16. "Compare the architectural features shown in Gothic and Renaissance cathedrals." -> image+table\n17. "Describe the moment of the moon landing and explain the mission details." -> paragraph+clip\n\nClassify the following query: {query}\nProvide only the category or categories combined with \'+\'.\n'
GATE_SPECS = {"gate_chen": dict(image_pos="last", retrieve=["true"], skip=["false"]),
              "gate_urag": dict(image_pos="none", retrieve=["paragraph", "document", "table", "image", "clip", "video"], skip=["no"])}
PROMPT_ID_V8 = hashlib.sha256("\x00".join([GATE_CHEN, GATE_URAG]).encode()).hexdigest()[:8]


def context_block(evidence, k=5, max_words=160):
    """Numbered evidence list: '[i] <title>: <text>' with per-passage word cap (E-VQA sections vary a lot)."""
    lines = []
    for i, e in enumerate(evidence[:k], 1):
        words = e["text"].split()
        text = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        lines.append(f"[{i}] {e['title']}: {text}")
    return "\n".join(lines)


def build(condition, rec, answer=None, k=5):
    """Text part of the user turn for a condition. The image is always attached by the model wrapper."""
    q = rec["question"]
    if rec.get("options") and condition in ("ne", "ev"):  # multiple choice (PREREG v7)
        if condition == "ne":
            return NE_MC.format(question=q, options=options_block(rec["options"]))
        return EV_MC.format(context=context_block(rec["evidence"], k), question=q, options=options_block(rec["options"]))
    if condition == "gate_chen":
        return GATE_CHEN.format(question=q)
    if condition == "gate_urag":
        return GATE_URAG.format(query=q)
    if condition in ("ne", "ne_sample", "think", "ne_tset"):
        return NE.format(question=q)
    if condition == "para":
        return PARA.format(question=q.lower().rstrip("?") + "?")
    if condition == "ev":
        return EV.format(context=context_block(rec["evidence"], k), question=q)
    if condition.startswith(("ev1_", "pv_")):  # one retrieved passage only: ev1_<rank> reads it, pv_<rank> verifies it
        one = [rec["evidence"][int(condition.split("_")[1]) - 1]]
        tpl = EV if condition.startswith("ev1_") else PV
        return tpl.format(context=context_block(one, 1, max_words=ONE_PASSAGE_WORDS), question=q)
    if condition.startswith("cv_"):  # one retrieved passage for the critic, as "# <page title>:\n<text>" (its card's format)
        e = rec["evidence"][int(condition.split("_")[1]) - 1]
        words = e["text"].split()
        text = " ".join(words[:ONE_PASSAGE_WORDS]) + (" ..." if len(words) > ONE_PASSAGE_WORDS else "")
        return CV_USER.format(question=q, passage=f"# {e['title']}:\n{text}")
    if condition == "ev_keep":
        return EV_KEEP.format(context=context_block(rec["evidence"], k), answer=answer, question=q)
    if condition == "oracle":
        return EV.format(context=context_block([{"title": "Reference", "text": rec["oracle_text"]}], 1,
                                               max_words=400), question=q)
    if condition in ("ent", "ent_sample"):
        return ENT
    if condition == "ptrue":
        return PTRUE.format(question=q, answer=answer)
    raise ValueError(condition)
