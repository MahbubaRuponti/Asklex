#!/usr/bin/env python3
"""
AskLex Legal Chatbot
====================
Bangladesh Law AI Assistant with Voice Support

This is your ONLY main file.
- Legal RAG (UKIL dataset + RRF hybrid retrieval)
- Cerebras LLM integration
- Voice input/output (optional)
- Interactive CLI

Works in: VS Code, Google Colab, Terminal
"""

import os
import sys
import re
import json
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import networkx as nx
import nltk
import torch
import numpy as np

from sentence_transformers import SentenceTransformer, util as st_util
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load .env file (API key, etc.)
load_dotenv()

# ===== VOICE LIBRARIES (Optional) =====

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    from pyttsx3 import init as tts_init
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    tts_init = None

try:
    from google.colab import output as colab_output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    colab_output = None

# ===== CEREBRAS CLIENT =====

try:
    from openai import OpenAI as CerebrasOpenAI
except Exception:
    CerebrasOpenAI = None

# ===== ENSURE NLTK DATA =====

try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

# ===== LOGGING =====

logger = logging.getLogger("AskLex")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ===== CONFIGURATION =====

HF_JSON_PATH = "hf://datasets/ciol-research/UKIL-DB-EN/acts_en.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CEREBRAS_MODEL = "llama3.1-8b"

TOP_K = 5
BM25_TOP_N = 20
EMBED_TOP_N = 10
MEMORY_WINDOW = 6
MAX_SECTION_CHARS = 1500
MAX_PROMPT_CHARS = 6000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RRF_K = 60.0

# ===== UTILITY FUNCTIONS =====

def clean_text(s: Any) -> str:
    """Clean and normalize text"""
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    return re.sub(r"\s+", " ", s).strip()

def safe_int(x, default: int = -1) -> int:
    """Safely convert to int"""
    try:
        return int(x)
    except Exception:
        return default

# ===== DATASET LOADING =====

def load_ukil_dataset(path: str) -> List[dict]:
    """Load UKIL legal dataset"""
    logger.info("Loading dataset from: %s", path)
    try:
        df = pd.read_json(path)
    except Exception as e:
        logger.warning("pandas.read_json failed (%s). Trying local JSON...", e)
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

# ===== SECTION INDEX (BM25 + Dense Embeddings) =====

class SectionIndex:
    """Index legal sections for hybrid retrieval"""

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

        # BM25 setup
        self.tokenized_corpus = [nltk.word_tokenize(clean_text(s["text"]).lower()) for s in self.sections]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Dense embeddings setup
        logger.info("Loading embedder (%s)...", embed_model_name)
        try:
            self.embedder = SentenceTransformer(embed_model_name, device="cuda" if device == "cuda" else "cpu")
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer: %s", e)
            raise

        logger.info("Encoding section texts...")
        texts = [clean_text(s["text"]) for s in self.sections]
        self.embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        logger.info("Embeddings shape: %s", tuple(self.embeddings.shape))

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

# ===== KNOWLEDGE GRAPH =====

def build_kg(acts: List[dict]) -> nx.DiGraph:
    """Build knowledge graph from legal acts"""
    G = nx.DiGraph()
    for a in acts:
        G.add_node(a["id"], name=a.get("name", ""), text=(a.get("text") or "")[:300])

    for a in acts:
        src = a["id"]
        for r in a.get("related_act", []) or []:
            try:
                tgt = int(r)
                if tgt != src:
                    G.add_edge(src, tgt, relation="related")
            except Exception:
                continue

    logger.info("Built KG: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
    return G

# ===== RRF HYBRID RETRIEVAL =====

def rrf_fusion_scores(
    section_index: SectionIndex,
    query: str,
    bm25_top_n: int = BM25_TOP_N,
    dense_top_n: int = EMBED_TOP_N,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion for hybrid retrieval"""
    N = section_index.N
    
    bm25_candidates = section_index.bm25_query(query, top_n=bm25_top_n)
    dense_ranked_all = section_index.dense_rank_all(query)
    dense_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_ranked_all)}
    
    bm25_rank_map = {}
    for rank, idx in enumerate(bm25_candidates):
        bm25_rank_map[idx] = rank + 1

    scores = []
    for idx in range(N):
        bm25_rank = bm25_rank_map.get(idx, bm25_top_n + 1 + (idx % 1000))
        dense_rank = dense_rank_map.get(idx, N + 1)
        rrf_score = (1.0 / (RRF_K + float(bm25_rank))) + (1.0 / (RRF_K + float(dense_rank)))
        scores.append((idx, rrf_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def hybrid_rrf_topk(section_index: SectionIndex, query: str, k: int = 5) -> List[int]:
    """Get top-k results using RRF"""
    fused = rrf_fusion_scores(section_index, query, bm25_top_n=BM25_TOP_N, dense_top_n=EMBED_TOP_N)
    return [idx for idx, _ in fused[:k]]

# ===== CONVERSATION MEMORY =====

class ConversationMemory:
    """Store conversation history and user profile"""

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

# ===== RAG PROMPT BUILDER =====

def build_rag_prompt(
    user_query: str,
    retrieved_sections: List[Tuple[int, float, dict]],
    section_index: SectionIndex,
    memory: ConversationMemory
) -> str:
    """Build RAG prompt with legal context"""
    convo_ctx = memory.get() if memory else ""
    profile = memory.get_profile() if memory else {}

    convo_block = f"Conversation history (recent):\n{convo_ctx}\n\n" if convo_ctx else "Conversation history: (none)\n\n"
    profile_block = "User profile:\n" + "\n".join([f"- {k}: {v}" for k, v in profile.items()]) + "\n\n" if profile else "User profile: (none)\n\n"

    prompt_parts = []
    prompt_parts.append("Legal excerpts from UKIL Database:\n")

    for i, sec in enumerate(retrieved_sections, 1):
        sec_idx, score, meta = sec
        header = f"[{i}] {meta.get('act_name', '')} | Section: {meta.get('section_name', '')}\n"
        text = clean_text(meta.get("text", ""))[:MAX_SECTION_CHARS]
        prompt_parts.append(header)
        prompt_parts.append(text + "\n\n")

    grounding_text = "".join(prompt_parts)
    total_prompt_body = convo_block + profile_block + "User question:\n" + user_query + "\n\n" + grounding_text

    if len(total_prompt_body) > MAX_PROMPT_CHARS:
        total_prompt_body = total_prompt_body[:MAX_PROMPT_CHARS] + "\n\n[TRUNCATED]"

    return total_prompt_body

# ===== CEREBRAS LLM =====

def init_cerebras_client() -> Optional[Any]:
    """Initialize Cerebras OpenAI-compatible client"""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        logger.warning("CEREBRAS_API_KEY not set - using fallback responses")
        return None

    if CerebrasOpenAI is None:
        logger.warning("OpenAI client not available")
        return None

    try:
        client = CerebrasOpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
        logger.info("✓ Cerebras client initialized")
        return client
    except Exception as e:
        logger.exception("Failed to initialize Cerebras: %s", e)
        return None

def call_cerebras_chat(client: Any, model: str, prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> str:
    """Call Cerebras LLM"""
    if client is None:
        raise RuntimeError("Cerebras client not initialized")

    messages = [
        {"role": "system", "content": "You are a legal assistant for Bangladesh. Base all responses on provided excerpts."},
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

# ===== VOICE INPUT =====

def get_voice_input() -> Optional[str]:
    """Get input from microphone"""
    if not SPEECH_RECOGNITION_AVAILABLE or IN_COLAB:
        logger.warning("Voice input not available")
        return None

    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("Listening... (speak now)")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)

        try:
            text = recognizer.recognize_google(audio)
            logger.info("Recognized: %s", text)
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.warning("Speech recognition error: %s", e)
            return None

    except Exception as e:
        logger.exception("Voice input error: %s", e)
        return None

# ===== VOICE OUTPUT =====

def speak_text(text: str) -> None:
    """Output text as speech"""
    if not PYTTSX3_AVAILABLE or IN_COLAB:
        return

    try:
        engine = tts_init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.exception("Voice output error: %s", e)

# ===== MAIN CHATBOT CLASS =====

class AskLexChatbot:
    """Legal chatbot with RAG and voice"""

    def __init__(
        self,
        section_index: SectionIndex,
        kg: nx.DiGraph,
        cerebras_client: Any
    ):
        self.section_index = section_index
        self.kg = kg
        self.cerebras_client = cerebras_client
        self.memory = ConversationMemory(max_len=MEMORY_WINDOW)

    def get_response(self, user_query: str, top_k: int = TOP_K) -> str:
        """Get response for query"""
        try:
            # Retrieve relevant sections
            retrieved_idx = hybrid_rrf_topk(self.section_index, user_query, k=top_k)
            retrieved = [
                (idx, 0.0, self.section_index.sections[idx])
                for idx in retrieved_idx
            ]

            # Build RAG prompt
            rag_prompt = build_rag_prompt(user_query, retrieved, self.section_index, self.memory)

            # Generate response
            if self.cerebras_client:
                response = call_cerebras_chat(
                    self.cerebras_client,
                    CEREBRAS_MODEL,
                    rag_prompt,
                    max_tokens=400,
                    temperature=0.2
                )
            else:
                # Fallback response
                top_act = self.section_index.sections[retrieved_idx[0]]["act_name"] if retrieved_idx else "the database"
                response = f"Based on {top_act}, here is the legal guidance. Please consult a qualified lawyer for specific advice."

            # Store in memory
            self.memory.add(user_query, response)

            return response

        except Exception as e:
            logger.exception("Error: %s", e)
            return f"Error: {str(e)}"

# ===== MAIN CLI =====

def main():
    """Interactive chatbot CLI"""
    print("\n" + "="*70)
    print("  AskLex: Bangladesh Legal AI Chatbot")
    print("="*70 + "\n")

    print("Environment:")
    print(f"  GPU: {torch.cuda.is_available()}")
    print(f"  Voice Input: {SPEECH_RECOGNITION_AVAILABLE}")
    print(f"  Voice Output: {PYTTSX3_AVAILABLE}")
    print(f"  Colab: {IN_COLAB}\n")

    # Initialize
    cerebras_client = init_cerebras_client()
    
    try:
        logger.info("Loading dataset...")
        acts = load_ukil_dataset(HF_JSON_PATH)
    except Exception as e:
        logger.error("Cannot load dataset: %s", e)
        return

    logger.info("Building indexes...")
    section_index = SectionIndex(acts, EMBED_MODEL_NAME, device=DEVICE)
    kg = build_kg(acts)

    # Create chatbot
    bot = AskLexChatbot(section_index, kg, cerebras_client)

    print("="*70)
    print("Commands:")
    print("  Ask: Type your legal question")
    print("  'record': Use microphone (local only)")
    print("  'profile': Set user profile (religion, marital status, etc.)")
    print("  'history': Show conversation")
    print("  'exit': Quit")
    print("="*70 + "\n")

    while True:
        try:
            print("\nYou: ", end="", flush=True)
            user_input = input().strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "history":
                history = bot.memory.get()
                print("\n--- Conversation ---")
                print(history if history else "(empty)")
                continue

            if user_input.lower() == "profile":
                print("Enter: key1: value1, key2: value2")
                profile_input = input().strip()
                try:
                    profile = {}
                    for part in profile_input.split(","):
                        if ":" in part:
                            k, v = part.split(":", 1)
                            profile[k.strip()] = v.strip()
                    bot.memory.set_profile(profile)
                    print("✓ Profile updated")
                except Exception as e:
                    print(f"Error: {e}")
                continue

            if user_input.lower() == "record":
                print("Recording...")
                voice_input = get_voice_input()
                if voice_input:
                    user_input = voice_input
                    print(f"Recognized: {user_input}")
                else:
                    print("Could not recognize. Type instead.")
                    continue

            # Get response
            print("\nProcessing...", flush=True)
            response = bot.get_response(user_input, top_k=TOP_K)

            print(f"\nAssistant: {response}")

            # Speak if available
            if PYTTSX3_AVAILABLE and not IN_COLAB:
                print("[Speaking...]")
                speak_text(response)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.exception("Error: %s", e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
