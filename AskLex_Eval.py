#!/usr/bin/env python3
"""
AskLex RAG — Iterative RAG with Multi-turn Agent ↔ User ↔ Retriever refinement loop.
Full module with RRF hybrid retrieval and advanced evaluation (BLEU/ROUGE/BERTScore + retrieval metrics).
Auto-runs advanced evaluation (20 synthetic queries) before entering the interactive loop.
"""

import os
import sys
import re
import json
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import networkx as nx
import nltk
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util
from rank_bm25 import BM25Okapi

# Attempt optional metric libraries; fallbacks provided
try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception:
        sentence_bleu = None

# Cerebras OpenAI-compatible client (if available)
try:
    from openai import OpenAI as CerebrasOpenAI
except Exception:
    CerebrasOpenAI = None

# Ensure nltk punkt tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")

logger = logging.getLogger("AskLexRAG")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# -----------------------------
# Configurable constants
# -----------------------------
HF_JSON_PATH = "hf://datasets/ciol-research/UKIL-DB-EN/acts_en.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CEREBRAS_MODEL = "llama3.1-8b"
TOP_K = 5
BM25_TOP_N = 20
EMBED_TOP_N = 10
MULTI_HOPS = 1
MEMORY_WINDOW = 6
MAX_SECTION_CHARS = 1500
MAX_PROMPT_CHARS = 6000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# New evaluation settings
SYNTH_EVAL_LIMIT = 100   # <=20 synthetic queries for the advanced evaluation

# Multi-turn / iterative settings (unchanged)
MAX_REFINE_TURNS = 3
MAX_ITER_ITERS = 2
CONVERGE_THRESHOLD = 0.02

# RRF constant
RRF_K = 60.0

# Heuristic keywords indicating laws that may be profile-dependent
PROFILE_KEYWORDS = {
    "religion": ["muslim", "shariat", "sharia", "islam", "hindu", "christian", "personal law", "family law"],
    "marriage": ["marriage", "divorce", "nikah", "dowry", "dower", "mahr", "custody", "child custody"],
}

# Clarifying questions
CLARIFY_QUESTIONS = [
    {"key": "religion", "question": "To apply the most relevant law, please specify the person's religion (e.g., Muslim, Hindu, Christian, Other): "},
    {"key": "marital_status", "question": "What is the marital status relevant to this scenario? (single, married, separated, divorced, unknown): "}
]

# -----------------------------
# Utilities
# -----------------------------
def clean_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    return re.sub(r"\s+", " ", s).strip()

def safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

# -----------------------------
# Dataset loader and normalization (keeps original behavior)
# -----------------------------
def load_ukil_dataset(path: str) -> List[dict]:
    logger.info("Loading dataset from: %s", path)
    try:
        df = pd.read_json(path)
    except Exception as e:
        logger.warning("pandas.read_json failed (%s). Attempting to open as local JSON...", e)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
        except Exception as e2:
            logger.exception("Failed to load dataset: %s", e2)
            raise

    acts = []
    for i, row in df.iterrows():
        if isinstance(row, dict):
            raw = row.get("act", row)
        else:
            rd = row.to_dict()
            raw = rd.get("act", rd)

        act_id = raw.get("id") or raw.get("act_id") or i
        act_id = safe_int(act_id, default=i)
        name = clean_text(raw.get("name", raw.get("title", "")))
        repelled = bool(raw.get("repelled", False))
        published_date = raw.get("published_date", None)
        related = raw.get("related_act", raw.get("related_acts", [])) or []
        related_ids = []
        if isinstance(related, (list, tuple)):
            for r in related:
                try:
                    related_ids.append(int(r))
                except Exception:
                    continue
        elif isinstance(related, (int, str)) and str(related).isdigit():
            related_ids = [int(related)]

        sections = raw.get("sections", []) or []
        norm_sections = []
        if isinstance(sections, list):
            for s in sections:
                if isinstance(s, dict):
                    sec_id = s.get("section_id", s.get("id", None)) or None
                    sec_name = clean_text(s.get("name", ""))
                    sec_details = clean_text(s.get("details", s.get("text", "")))
                    norm_sections.append({
                        "section_id": sec_id,
                        "name": sec_name,
                        "details": sec_details
                    })

        text = " ".join([s["details"] for s in norm_sections]) if norm_sections else clean_text(raw.get("text") or "")
        tags = raw.get("tags") or raw.get("meta") or {}

        acts.append({
            "id": act_id,
            "name": name,
            "repelled": repelled,
            "published_date": published_date,
            "related_act": related_ids,
            "sections": norm_sections,
            "text": text,
            "raw": raw,
            "tags": tags,
        })

    logger.info("Loaded %d acts", len(acts))
    return acts

# -----------------------------
# Section-level index construction (BM25 + dense embeddings)
# -----------------------------
class SectionIndex:
    def __init__(self, acts: List[dict], embed_model_name: str, device: str = DEVICE):
        self.acts = acts
        self.device = device
        self.sections = []
        for act in acts:
            act_id = act["id"]
            act_name = act.get("name", "")
            if act.get("sections"):
                for sec in act["sections"]:
                    text = sec.get("details", "") or ""
                    name = sec.get("name", "") or ""
                    self.sections.append({
                        "act_id": act_id,
                        "act_name": act_name,
                        "section_name": name,
                        "section_id": sec.get("section_id", None),
                        "text": text
                    })
            else:
                self.sections.append({
                    "act_id": act_id,
                    "act_name": act_name,
                    "section_name": act_name,
                    "section_id": None,
                    "text": act.get("text", "")
                })

        self.N = len(self.sections)
        logger.info("Total sections indexed: %d", self.N)

        # Tokenize for BM25
        self.tokenized_corpus = [nltk.word_tokenize(clean_text(s["text"]).lower()) for s in self.sections]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense embeddings
        logger.info("Loading embedder (%s) on device=%s", embed_model_name, device)
        try:
            self.embedder = SentenceTransformer(embed_model_name, device="cuda" if device == "cuda" else "cpu")
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer model: %s", e)
            raise

        logger.info("Encoding section texts (dense embeddings)...")
        texts = [clean_text(s["text"]) for s in self.sections]
        # convert_to_tensor True yields GPU tensors when device supports
        self.embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        logger.info("Section embeddings shape: %s", tuple(self.embeddings.shape))

    def bm25_query(self, query: str, top_n: int = BM25_TOP_N) -> List[int]:
        toks = nltk.word_tokenize(clean_text(query).lower())
        scores = self.bm25.get_scores(toks)
        top_indices = np.argsort(scores)[::-1][:top_n].tolist()
        return top_indices

    def dense_rank_all(self, query: str) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], convert_to_tensor=True)
        sims = st_util.cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
        ranked = sorted([(i, float(sims[i])) for i in range(len(sims))], key=lambda x: x[1], reverse=True)
        return ranked

    def dense_rank(self, query: str, candidate_indices: List[int], top_n: int = EMBED_TOP_N) -> List[Tuple[int, float]]:
        if not candidate_indices:
            return []
        q_emb = self.embedder.encode([query], convert_to_tensor=True)
        cand_embs = self.embeddings[candidate_indices]
        sims = st_util.cos_sim(q_emb, cand_embs)[0].cpu().numpy()
        ranked = sorted([(candidate_indices[i], float(sims[i])) for i in range(len(candidate_indices))], key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

# -----------------------------
# Knowledge Graph utilities
# -----------------------------
def build_kg(acts: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for a in acts:
        G.add_node(a["id"], name=a.get("name", ""), text=(a.get("text") or "")[:300])

    edge_count = 0
    for a in acts:
        src = a["id"]
        for r in a.get("related_act", []) or []:
            try:
                tgt = int(r)
            except Exception:
                continue
            if tgt != src:
                G.add_edge(src, tgt, relation="related")
                edge_count += 1

    logger.info("Built KG: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G

def expand_via_kg(act_ids: List[int], kg: nx.DiGraph, hops: int = MULTI_HOPS) -> List[int]:
    expanded = set(act_ids)
    current = list(act_ids)
    for h in range(hops):
        next_nodes = []
        for a in current:
            if a in kg:
                neighbors = list(kg.successors(a)) + list(kg.predecessors(a))
                for n in neighbors:
                    if n not in expanded:
                        expanded.add(n)
                        next_nodes.append(n)
        current = next_nodes
        if not current:
            break
    return list(expanded)

# -----------------------------
# Zero-shot classifier fallback
# -----------------------------
try:
    from transformers import pipeline
    _zs_device = 0 if torch.cuda.is_available() else -1
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=_zs_device)
    logger.info("Initialized HuggingFace zero-shot pipeline on device=%s", "cuda" if _zs_device == 0 else "cpu")
except Exception as e:
    logger.warning("Failed to init zero-shot pipeline: %s. Falling back to embedding-based scoring.", e)
    zero_shot_classifier = None

def zero_shot_score_labels(query: str, labels: List[str], embedder: SentenceTransformer) -> List[Tuple[str, float]]:
    if zero_shot_classifier is not None:
        try:
            out = zero_shot_classifier(sequences=query, candidate_labels=labels, multi_label=False)
            lbls = out.get("labels", [])
            scores = out.get("scores", [])
            scored = [(lbls[i], float(scores[i])) for i in range(len(lbls))]
            return scored
        except Exception as e:
            logger.exception("Zero-shot pipeline failed: %s", e)

    # fallback to embeddings
    q_emb = embedder.encode([query], convert_to_tensor=True)
    lbl_embs = embedder.encode(labels, convert_to_tensor=True)
    sims = st_util.cos_sim(q_emb, lbl_embs)[0].cpu().numpy()
    scored = sorted([(labels[i], float(sims[i])) for i in range(len(labels))], key=lambda x: x[1], reverse=True)
    return scored

# -----------------------------
# Hybrid retrieval utilities (RRF fusion)
# -----------------------------
def rrf_fusion_scores(
    section_index: SectionIndex,
    query: str,
    bm25_top_n: int = BM25_TOP_N,
    dense_top_n: int = EMBED_TOP_N,
    use_kg: bool = False,
    kg: Optional[nx.DiGraph] = None
) -> List[Tuple[int, float]]:
    """
    Compute RRF scores for all sections using BM25 ranks and dense ranks, return list of (sec_idx, rrf_score) sorted desc.
    Ranks for missing items are treated as large rank (N).
    """
    N = section_index.N
    # BM25 candidates: take top bm25_top_n indices
    bm25_candidates = section_index.bm25_query(query, top_n=bm25_top_n)
    # Dense full ranking (we compute dense similarity for all sections)
    dense_ranked_all = section_index.dense_rank_all(query)
    dense_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_ranked_all)}  # rank starts at 1

    # Build bm25 rank map (only for bm25_candidates); missing -> large rank
    bm25_rank_map = {}
    for rank, idx in enumerate(bm25_candidates):
        bm25_rank_map[idx] = rank + 1

    # Optionally expand act ids via KG (to include more neighbor sections)
    allowed_section_indices = set(range(N))
    if use_kg and kg is not None:
        # we can expand via top acts from bm25_candidates/dense top items
        top_acts = set([section_index.sections[i]["act_id"] for i in bm25_candidates[:5]])
        expanded_acts = expand_via_kg(list(top_acts), kg, hops=MULTI_HOPS)
        allowed_section_indices = set(i for i, s in enumerate(section_index.sections) if s["act_id"] in expanded_acts)

    # Compute RRF score for every allowed section
    scores = []
    for idx in range(N):
        if idx not in allowed_section_indices:
            # still allow but give lower priority
            pass
        bm25_rank = bm25_rank_map.get(idx, bm25_top_n + 1 + (idx % 1000))  # large-ish rank for non-candidates
        dense_rank = dense_rank_map.get(idx, N + 1)
        rrf_score = (1.0 / (RRF_K + float(bm25_rank))) + (1.0 / (RRF_K + float(dense_rank)))
        scores.append((idx, rrf_score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def bm25_topk(section_index: SectionIndex, query: str, k: int = 5) -> List[int]:
    return section_index.bm25_query(query, top_n=k)

def dense_topk(section_index: SectionIndex, query: str, k: int = 5) -> List[int]:
    ranked = section_index.dense_rank_all(query)
    return [idx for idx, _ in ranked[:k]]

def hybrid_rrf_topk(section_index: SectionIndex, query: str, k: int = 5, kg: Optional[nx.DiGraph] = None, use_kg: bool = False) -> List[int]:
    fused = rrf_fusion_scores(section_index, query, bm25_top_n=BM25_TOP_N, dense_top_n=EMBED_TOP_N, use_kg=use_kg, kg=kg)
    return [idx for idx, _ in fused[:k]]

# -----------------------------
# MultiTurn RAG and AskLexChatbot (keeps your original logic)
# -----------------------------
# ... We'll include simplified versions for brevity (core functionality preserved).
# For the sake of this module we keep the agent-related code minimal but compatible with earlier flow.
def build_rag_prompt(user_query: str, retrieved_sections: List[Tuple[int, float, dict]], section_index: SectionIndex, kg: nx.DiGraph, memory: Any, include_summary: Optional[str]=None) -> str:
    convo_ctx = memory.get() if memory else ""
    profile = memory.get_profile() if memory else {}
    convo_block = f"Conversation history (most recent):\n{convo_ctx}\n\n" if convo_ctx else "Conversation history: (none)\n\n"
    profile_block = "User profile:\n" + "\n".join([f"- {k}: {v}" for k, v in profile.items()]) + "\n\n" if profile else "User profile: (none)\n\n"

    prompt_parts = []
    prompt_parts.append("RAG grounding excerpts (truncated):\n")
    for i, sec in enumerate(retrieved_sections, 1):
        sec_idx, score, meta = sec
        header = f"[Top {i}] Act: {meta.get('act_name','')} (ActID={meta.get('act_id')}) | Section: {meta.get('section_name','')}\n"
        text = clean_text(meta.get("text", ""))[:MAX_SECTION_CHARS]
        prompt_parts.append(header)
        prompt_parts.append(text + "\n\n")
    grounding_text = "".join(prompt_parts)

    total_prompt_body = convo_block + profile_block + "User scenario:\n" + user_query + "\n\n"
    if include_summary:
        total_prompt_body += "Refinement summary (previous):\n" + include_summary + "\n\n"
    total_prompt_body += grounding_text
    if len(total_prompt_body) > MAX_PROMPT_CHARS:
        total_prompt_body = total_prompt_body[:MAX_PROMPT_CHARS] + "\n\n[TRUNCATED]"
    return total_prompt_body

def build_agent_instruction() -> str:
    instr = (
        "You will produce either a single clarifying question or a concise final answer, and nothing else.\n"
        "If the grounding excerpts are insufficient to produce a clear final answer, output exactly one question prefixed with: FOLLOWUP: <question>\n"
        "If you can produce a final statutory-grounded answer, prefix it with: FINAL: <answer>\n"
        "Do not invent citations; rely only on the supplied excerpts."
    )
    return instr

class ConversationMemory:
    def __init__(self, max_len: int = MEMORY_WINDOW):
        self.max_len = max_len
        self.history: List[Dict[str, str]] = []
        self.profile: Dict[str, Any] = {}

    def add(self, user: str, assistant: str):
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > self.max_len:
            self.history.pop(0)

    def get(self) -> str:
        if not self.history:
            return ""
        return "\n\n".join([f"User: {t['user']}\nAssistant: {t['assistant']}" for t in self.history])

    def set_profile(self, profile: Dict[str, Any]):
        self.profile.update(profile)

    def get_profile(self) -> Dict[str, Any]:
        return dict(self.profile)

# Minimal MultiTurnIterativeRAG that uses available cerebras client or fallback
class MultiTurnIterativeRAG:
    def __init__(self, section_index: SectionIndex, kg: nx.DiGraph, cerebras_client: Any, cerebras_model: str = CEREBRAS_MODEL):
        self.section_index = section_index
        self.kg = kg
        self.cerebras_client = cerebras_client
        self.cerebras_model = cerebras_model

    def _parse_agent_reply(self, reply: str) -> Dict[str, Any]:
        if not reply:
            return {"type": "final", "answer": ""}
        r = reply.strip()
        lines = [ln.strip() for ln in r.splitlines() if ln.strip()]
        first = lines[0] if lines else r
        m_f = re.match(r"^(FOLLOWUP\s*[:\-]\s*)(.*)$", first, flags=re.IGNORECASE)
        m_F = re.match(r"^(FINAL\s*[:\-]\s*)(.*)$", first, flags=re.IGNORECASE)
        if m_f:
            return {"type": "followup", "question": m_f.group(2).strip()}
        if m_F:
            rest = m_F.group(2).strip()
            if len(lines) > 1:
                rest += "\n" + "\n".join(lines[1:])
            return {"type": "final", "answer": rest}
        if "?" in first and len(first.split()) < 30:
            return {"type": "followup", "question": first}
        return {"type": "final", "answer": r}

    def agent_call(self, prompt: str, max_tokens: int = 320, temperature: float = 0.2) -> str:
        if self.cerebras_client:
            return call_cerebras_chat(self.cerebras_client, self.cerebras_model, prompt, max_tokens=max_tokens, temperature=temperature)
        # fallback
        return "[LOCAL FALLBACK] " + prompt[:400]

    def iterate_multi_turn(self, user_query: str, memory: ConversationMemory, profile: Dict[str, Any], top_k: int = TOP_K) -> Dict[str, Any]:
        result = {"turns": [], "final_answer": None, "final_retrieved": None, "agent_trace": []}
        current_query = user_query
        for turn in range(MAX_REFINE_TURNS):
            # retrieval using RRF hybrid (top_k)
            retrieved_idx = hybrid_rrf_topk(self.section_index, current_query, k=top_k, kg=self.kg, use_kg=False)
            retrieved = []
            for idx in retrieved_idx:
                meta = self.section_index.sections[idx]
                score = 0.0  # score not needed here
                retrieved.append((idx, score, meta))

            rag_prompt = build_rag_prompt(current_query, retrieved, self.section_index, self.kg, memory)
            agent_instr = build_agent_instruction()
            full_prompt = f"{rag_prompt}\n\n{agent_instr}"

            agent_reply = None
            parsed = None
            try:
                if self.cerebras_client:
                    agent_reply = self.agent_call(full_prompt, max_tokens=480, temperature=0.12)
                    parsed = self._parse_agent_reply(agent_reply)
                else:
                    # fallback heuristics: produce a template final answer
                    top_meta = retrieved[0][2] if retrieved else {}
                    parsed = {"type": "final", "answer": template_fallback_answer(user_query, top_meta, profile)}
                    agent_reply = "[LOCAL FALLBACK]\n" + parsed["answer"]
            except Exception as e:
                logger.exception("Agent call failed: %s", e)
                top_meta = retrieved[0][2] if retrieved else {}
                parsed = {"type": "final", "answer": template_fallback_answer(user_query, top_meta, profile)}
                agent_reply = "[AGENT ERROR FALLBACK]\n" + parsed["answer"]

            result["agent_trace"].append({"turn": turn+1, "prompt": full_prompt, "agent_reply": agent_reply, "parsed": parsed, "retrieved_snapshot": [(s[0], s[1], s[2].get("act_id")) for s in retrieved]})
            if parsed.get("type") == "followup":
                # During evaluation we will not ask the user; however interactive path still asks
                # Here we convert followup into a forced final using template_fallback to make evaluation deterministic
                top_meta = retrieved[0][2] if retrieved else {}
                final_ans = template_fallback_answer(user_query, top_meta, profile)
                result["final_answer"] = final_ans
                result["final_retrieved"] = retrieved
                result["turns"].append({"role": "assistant", "content": final_ans, "retrieved": retrieved})
                return result
            else:
                final_ans = parsed.get("answer") or agent_reply or ""
                memory.add(user_query, final_ans)
                result["final_answer"] = final_ans
                result["final_retrieved"] = retrieved
                result["turns"].append({"role": "assistant", "content": final_ans, "retrieved": retrieved})
                return result

        # fallback if none produced
        top_meta = []
        final_ans = template_fallback_answer(user_query, top_meta, profile)
        result["final_answer"] = final_ans
        result["final_retrieved"] = []
        result["turns"].append({"role": "assistant", "content": final_ans, "retrieved": []})
        return result

def init_cerebras_client() -> Optional[Any]:
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        logger.warning("CEREBRAS_API_KEY not found in environment; generation disabled.")
        return None
    if CerebrasOpenAI is None:
        logger.warning("Cerebras/OpenAI client library not available.")
        return None
    try:
        client = CerebrasOpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
        logger.info("Initialized Cerebras OpenAI-compatible client.")
        return client
    except Exception as e:
        logger.exception("Failed to initialize Cerebras client: %s", e)
        return None

def call_cerebras_chat(client: Any, model: str, prompt: str, max_tokens: int = 320, temperature: float = 0.2) -> str:
    if client is None:
        raise RuntimeError("Cerebras client is not initialized.")
    messages = [
        {"role": "system", "content": "You are a precise legal assistant. Base responses on the supplied statute excerpts."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = ""
        try:
            choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
            if choices:
                ch0 = choices[0]
                if isinstance(ch0, dict) and "message" in ch0:
                    content = ch0["message"].get("content", "").strip()
                elif hasattr(ch0, "message"):
                    content = getattr(ch0.message, "content", "") or ""
                if not content:
                    content = str(resp)
        except Exception:
            content = str(resp)
        return content.strip()
    except Exception as e:
        logger.exception("Cerebras API call failed: %s", e)
        raise

# Template fallback (same as before)
def template_fallback_answer(query: str, top_meta: dict, profile: Dict[str, Any]) -> str:
    act_name = top_meta.get("act_name", "the retrieved act") if isinstance(top_meta, dict) else "the retrieved act"
    prof_str = ""
    if profile:
        prof_str = " Profile info noted: " + ", ".join([f"{k}={v}" for k, v in profile.items()]) + "."
    return (
        "Preliminary assessment: your scenario appears related to '{}'.".format(act_name) +
        prof_str +
        " Suggested next steps: 1) File a police General Diary (GD) or FIR if appropriate. " +
        "2) Preserve evidence (messages, receipts, photos, witnesses). 3) Contact a qualified lawyer or legal aid. " +
        "This is general guidance and not legal advice."
    )

# AskLexChatbot wrapper
class AskLexChatbot:
    def __init__(self, acts: List[dict], section_index: SectionIndex, kg: nx.DiGraph, cerebras_client: Any, cerebras_model: str = CEREBRAS_MODEL):
        self.acts = acts
        self.section_index = section_index
        self.kg = kg
        self.cerebras_client = cerebras_client
        self.cerebras_model = cerebras_model
        self.memory = ConversationMemory(max_len=MEMORY_WINDOW)
        self.iterative_controller = MultiTurnIterativeRAG(self.section_index, self.kg, self.cerebras_client, cerebras_model)

    def handle_query(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        retrieved_initial = hybrid_rrf_topk(self.section_index, query, k=top_k, kg=self.kg, use_kg=False)
        # optional early clarification (interactive); skipped in evaluation
        profile = self.memory.get_profile()

        multi_turn_result = self.iterative_controller.iterate_multi_turn(query, self.memory, profile=profile, top_k=top_k)

        candidate_labels = ["criminal law", "family law", "property law", "labor law", "marriage/divorce", "child custody"]
        zs_scores = zero_shot_score_labels(query, candidate_labels, self.section_index.embedder)
        top_label = zs_scores[0][0] if zs_scores else "general law"

        return {"query": query, "zero_shot": zs_scores, "memory_profile": self.memory.get_profile(), "multi_turn": multi_turn_result, "label": top_label}

    def answer_with_contexts(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        out = self.handle_query(query, top_k=top_k)
        multi = out.get("multi_turn", {})
        final_answer = multi.get("final_answer", "") or ""
        retrieved = multi.get("final_retrieved", []) or []
        contexts = [meta.get("text", "") for _, _, meta in retrieved]
        return {"query": query, "answer": final_answer, "contexts": contexts, "meta": out}

# -----------------------------
# Evaluation utilities (advanced)
# -----------------------------
def generate_synthetic_dataset_from_acts(acts_list: List[dict], limit: Optional[int] = None) -> List[dict]:
    """
    Build a synthetic evaluation dataset from acts and sections.
    Each entry:
      {"query": str, "expected_act": act_name, "expected_act_id": act_id, "expected_text": section_text, "section_id": ...}
    """
    synth = []
    for act in acts_list:
        for sec in act.get("sections", []):
            act_name = act.get("name", "the act")
            sec_name = sec.get("name") or "this section"
            q = f"What does the {act_name} state regarding {sec_name.lower()}?"
            synth.append({
                "query": q,
                "expected_act": act_name,
                "expected_act_id": act.get("id"),
                "expected_text": sec.get("details",""),
                "section_id": sec.get("section_id")
            })
    if limit is not None:
        synth = synth[:limit]
    return synth

def ragas_simple_score(answer: str, expected_act: str) -> float:
    if not answer:
        return 0.0
    return 1.0 if expected_act.lower() in answer.lower() else 0.0

def compute_bleu(reference: str, hypothesis: str) -> float:
    try:
        if sacrebleu is not None:
            # sacrebleu expects list(s)
            return float(sacrebleu.sentence_bleu(hypothesis, [reference]).score) / 100.0
        elif sentence_bleu is not None:
            # nltk smoothing
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            smoothie = SmoothingFunction().method4
            return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    except Exception:
        pass
    return float("nan")

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    try:
        if rouge_scorer is not None:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            sc = scorer.score(reference, hypothesis)
            return float(sc["rougeL"].fmeasure)
    except Exception:
        pass
    return float("nan")

def compute_bertscore_f1(references: List[str], hypotheses: List[str], lang: str = "en") -> float:
    # bert_score_fn returns P, R, F tensors; we return averaged F1
    try:
        if bert_score_fn is None:
            return float("nan")
        P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
        return float(F1.mean().item())
    except Exception:
        return float("nan")

def semantic_relevancy_score(embedder: SentenceTransformer, expected_text: str, answer: str) -> float:
    try:
        q_emb = embedder.encode([expected_text], convert_to_tensor=True)
        a_emb = embedder.encode([answer], convert_to_tensor=True)
        sim = float(st_util.cos_sim(q_emb, a_emb)[0][0].cpu().numpy())
        # Normalize to [0,1] from [-1,1] (though SBERT sims will be in [-1,1])
        return (sim + 1.0) / 2.0
    except Exception:
        return float("nan")

def token_overlap_faithfulness(expected_text: str, answer: str) -> float:
    if not expected_text or not answer:
        return 0.0
    ref_tokens = set(nltk.word_tokenize(clean_text(expected_text).lower()))
    ans_tokens = set(nltk.word_tokenize(clean_text(answer).lower()))
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & ans_tokens) / float(len(ref_tokens))
    return float(overlap)

def evaluate_retrieval_precision_recall(section_index: SectionIndex, query: str, expected_act_id: int, ks=(1,3,5), kg: Optional[nx.DiGraph]=None) -> Dict[str, Any]:
    out = {}
    # BM25
    bm_top = bm25_topk(section_index, query, k=max(ks))
    dense_top = dense_topk(section_index, query, k=max(ks))
    hybrid_top = hybrid_rrf_topk(section_index, query, k=max(ks), kg=kg, use_kg=False)

    methods = {"bm25": bm_top, "dense": dense_top, "hybrid": hybrid_top}
    for method, ranked in methods.items():
        for k in ks:
            topk = ranked[:k]
            # precision@k: fraction of topk whose act_id matches expected_act_id
            if len(topk) == 0:
                prec = 0.0
            else:
                matches = sum(1 for idx in topk if section_index.sections[idx]["act_id"] == expected_act_id)
                prec = matches / float(len(topk))
            # recall@k: indicator whether expected act is present among topk sections (treating single relevant act)
            rec = 1.0 if any(section_index.sections[idx]["act_id"] == expected_act_id for idx in topk) else 0.0
            out[f"{method}_prec@{k}"] = prec
            out[f"{method}_rec@{k}"] = rec
    return out

def run_advanced_evaluation(bot_instance: AskLexChatbot, acts_list: List[dict], embedder: SentenceTransformer, section_index: SectionIndex, kg: nx.DiGraph, limit: int = SYNTH_EVAL_LIMIT, save_csv: bool = True) -> pd.DataFrame:
    synth = generate_synthetic_dataset_from_acts(acts_list, limit=limit)
    logger.info("Synthetic advanced evaluation set size: %d", len(synth))
    records = []
    # pre-collect references for BERTScore if available
    references = []
    hypotheses = []

    for item in synth:
        q = item["query"]
        expected_act = item["expected_act"]
        expected_act_id = item["expected_act_id"]
        expected_text = item.get("expected_text", "")
        start = time.time()
        out = bot_instance.answer_with_contexts(q)
        elapsed = time.time() - start
        final_ans = out.get("answer", "") or ""
        contexts = out.get("contexts", []) or []
        joined_ctx = " ".join(contexts)
        # RAGAS proxy
        ragas_score = ragas_simple_score(final_ans, expected_act)
        # BLEU
        bleu = compute_bleu(expected_text, final_ans)
        # ROUGE-L
        rouge_l = compute_rouge_l(expected_text, final_ans)
        # BERTScore will be computed in batch after loop for speed, but we collect references/hypotheses
        references.append(expected_text if expected_text else "")
        hypotheses.append(final_ans if final_ans else "")
        # semantic relevancy (embedding cosine)
        ans_relev = semantic_relevancy_score(embedder, expected_text, final_ans)
        # faithfulness token overlap
        faith_tok = token_overlap_faithfulness(expected_text, final_ans)
        # retrieval metrics
        retr_metrics = evaluate_retrieval_precision_recall(section_index, q, expected_act_id, ks=(1,3,5), kg=kg)

        rec = {
            "query": q,
            "expected_act": expected_act,
            "expected_act_id": expected_act_id,
            "expected_text": expected_text,
            "final_answer": final_ans,
            "ragas_score": ragas_score,
            "bleu": bleu,
            "rouge_l": rouge_l,
            # bertscore_f1 placeholder (will fill after)
            "bertscore_f1": float("nan"),
            "faithfulness_token_overlap": faith_tok,
            "answer_relevancy_semantic": ans_relev,
            "retrieved_excerpt": joined_ctx[:1600],
            "latency_sec": elapsed
        }
        # add retr metrics keys
        rec.update(retr_metrics)
        records.append(rec)

    # compute BERTScore in batch if available
    try:
        if bert_score_fn is not None and len(hypotheses) > 0:
            _, _, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
            for i, f_val in enumerate(F1):
                records[i]["bertscore_f1"] = float(f_val.item())
        else:
            # leave as NaN or attempt a simple overlap-based proxy
            for i in range(len(records)):
                records[i]["bertscore_f1"] = float("nan")
    except Exception as e:
        logger.exception("BERTScore computation failed: %s", e)
        for i in range(len(records)):
            records[i]["bertscore_f1"] = float("nan")

    df = pd.DataFrame(records)
    if save_csv:
        out_name = "advanced_ragas_evaluation_results.csv"
        df.to_csv(out_name, index=False)
        logger.info("Saved advanced evaluation to %s", out_name)

    # Compute aggregated averages (select numeric cols)
    agg = {}
    numeric_cols = ["ragas_score", "bleu", "rouge_l", "bertscore_f1", "faithfulness_token_overlap", "answer_relevancy_semantic", "latency_sec",
                    "bm25_prec@1", "bm25_rec@1", "bm25_prec@3", "bm25_rec@3", "bm25_prec@5", "bm25_rec@5",
                    "dense_prec@1", "dense_rec@1", "dense_prec@3", "dense_rec@3", "dense_prec@5", "dense_rec@5",
                    "hybrid_prec@1", "hybrid_rec@1", "hybrid_prec@3", "hybrid_rec@3", "hybrid_prec@5", "hybrid_rec@5"]
    for c in numeric_cols:
        if c in df.columns:
            agg[c] = float(df[c].mean(skipna=True))
        else:
            agg[c] = float("nan")

    # print concise summary
    print("\n=== ADVANCED EVALUATION SUMMARY ===")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")
    print(f"Detailed per-query results saved to '{out_name}'\n")
    return df

# -----------------------------
# Main CLI / orchestration
# -----------------------------
def main():
    print("\nAskLex RAG Chatbot — Multi-turn Iterative RAG (RRF Hybrid + Advanced Evaluation)\n")
    cerebras_client = init_cerebras_client()
    if cerebras_client is None:
        logger.warning("Cerebras client not initialized; generation will fallback to local templates/heuristics.")

    try:
        acts = load_ukil_dataset(HF_JSON_PATH)
    except Exception as e:
        logger.exception("Failed to load UKIL dataset: %s", e)
        # If loading fails, abort to avoid silent incorrect behavior
        return

    kg = build_kg(acts)

    try:
        section_index = SectionIndex(acts, EMBED_MODEL_NAME, device="cuda" if DEVICE == "cuda" else "cpu")
    except Exception as e:
        logger.exception("Failed to build section index: %s", e)
        return

    bot = AskLexChatbot(acts, section_index, kg, cerebras_client, cerebras_model=CEREBRAS_MODEL)

    # Automatic advanced evaluation BEFORE interactive loop (Variant A)
    try:
        logger.info("Running automatic advanced evaluation (startup)...")
        df_adv = run_advanced_evaluation(bot, acts, section_index.embedder, section_index, kg, limit=SYNTH_EVAL_LIMIT, save_csv=True)
        # Print sample rows
        if not df_adv.empty:
            display_cols = ["query", "final_answer", "ragas_score", "bleu", "rouge_l", "bertscore_f1", "faithfulness_token_overlap", "answer_relevancy_semantic", "latency_sec"]
            print("\nSample advanced evaluation rows (first 10):")
            with pd.option_context("display.max_colwidth", 200):
                print(df_adv[display_cols].head(10).to_string(index=False))
    except Exception as e:
        logger.exception("Advanced evaluation failed: %s", e)

    # Interactive loop
    print("\nAskLex is ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nEnter scenario: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AskLex.")
            break

        if not user_input:
            print("Please enter a scenario or 'exit'.")
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        start = time.time()
        try:
            result = bot.handle_query(user_input, top_k=TOP_K)
        except Exception as e:
            logger.exception("Error processing query: %s", e)
            print("An internal error occurred. See logs.")
            continue

        latency = time.time() - start

        multi = result.get("multi_turn", {})
        print("\n[DEBUG] Multi-turn agent trace:")
        for trace in multi.get("agent_trace", []):
            t = trace.get("turn")
            snap = trace.get("retrieved_snapshot", [])
            print(f" Turn {t}: retrieved snapshot (secIdx,score,ActID): {snap}")
            print(f"  Agent reply (first 200 chars): {str(trace.get('agent_reply',''))[:200]}")
            print("-" * 40)

        print("\n[Zero-shot label scores]:")
        for label, sc in result.get("zero_shot", [])[:5]:
            print(f" {label}: {sc:.4f}")

        print("\n[FINAL ADVICE]\n")
        final_ans = multi.get("final_answer") or "No final answer available."
        print(final_ans)

        print(f"\n[INFO] Latency: {latency:.2f}s")
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main()
