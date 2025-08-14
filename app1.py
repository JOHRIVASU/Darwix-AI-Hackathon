# app.py — Empathetic Code Reviewer (Multi‑Language Support)
# Run: streamlit run app.py
import os, re, difflib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import autopep8
import textstat

# Optional extras
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# Multi-language complexity (supports C/C++/Java/JS/TS/Go/Ruby/Python/etc.)
try:
    import lizard
    _LIZARD_OK = True
except Exception:
    _LIZARD_OK = False

# Language detection / code fences
try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    _PYGMENTS_OK = True
except Exception:
    _PYGMENTS_OK = False


# ---------------------------- Language maps ----------------------------
LANG_RESOURCES = {
    "python": "PEP 8 – https://peps.python.org/pep-0008/",
    "javascript": "JavaScript Guide + ESLint – https://eslint.org/docs/latest/",
    "typescript": "TypeScript Handbook – https://www.typescriptlang.org/docs/",
    "java": "Google Java Style Guide – https://google.github.io/styleguide/javaguide.html",
    "c++": "Google C++ Style Guide – https://google.github.io/styleguide/cppguide.html",
    "c": "C style & MISRA (general) – https://en.wikipedia.org/wiki/MISRA_C",
    "go": "Effective Go – https://go.dev/doc/effective_go",
    "ruby": "Ruby Style Guide – https://rubystyle.guide/",
    "c#": "C# Guidelines – https://learn.microsoft.com/dotnet/csharp/fundamentals/coding-style/coding-conventions",
    "php": "PHP Standard Recommendations (PSR) – https://www.php-fig.org/psr/",
    "rust": "Rust Style / Clippy – https://doc.rust-lang.org/1.0.0/style/",
    "kotlin": "Kotlin Style Guide – https://developer.android.com/kotlin/style-guide",
    "swift": "Swift API Design Guidelines – https://www.swift.org/documentation/api-design-guidelines/",
}

# Mappings between pygments names and code fence tags
FENCE_ALIASES = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "java": "java",
    "c++": "cpp",
    "cpp": "cpp",
    "c": "c",
    "go": "go",
    "golang": "go",
    "ruby": "ruby",
    "rb": "ruby",
    "c#": "csharp",
    "csharp": "csharp",
    "php": "php",
    "rust": "rust",
    "kotlin": "kotlin",
    "swift": "swift",
}

LANG_LIST = ["Auto‑detect"] + [
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "Go", "Ruby", "C#", "PHP", "Rust", "Kotlin", "Swift"
]


# ---------------------------- Helpers ----------------------------
def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-\_ ]+", "", s).strip().lower().replace(" ", "-")
    return re.sub(r"-{2,}", "-", s)[:80] or "report"

def code_fence_lang(lang_key: str) -> str:
    return FENCE_ALIASES.get(lang_key.lower(), "text")

def guess_language(code: str, manual_choice: str) -> str:
    if manual_choice and manual_choice.lower() != "auto‑detect":
        return manual_choice.lower()
    if _PYGMENTS_OK:
        try:
            lex = guess_lexer(code)
            name = lex.name.lower()
            # normalize common cases
            if "javascript" in name: return "javascript"
            if "typescript" in name: return "typescript"
            if name in ["python", "java", "c++", "c", "go", "ruby", "php", "rust", "kotlin", "swift", "c#", "csharp"]:
                return name
        except ClassNotFound:
            pass
    # heuristic fallback
    if re.search(r"package\s+main|fmt\.Print", code): return "go"
    if re.search(r"(#include\s+<)|(::)", code): return "c++"
    if re.search(r"class\s+[A-Z]\w+|public\s+static\s+void\s+main", code): return "java"
    if re.search(r"def\s+\w+\(", code): return "python"
    if re.search(r"function\s+\w+\(|=>", code): return "javascript"
    return "python"

def unified_diff(a: str, b: str, fromfile="original", tofile="improved") -> str:
    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="")
    return "\n".join(diff) or "(No changes)"

def metrics_for_any(code: str, lang_key: str) -> Dict:
    # If Python, also compute readability and long lines
    if _LIZARD_OK:
        # give a pseudo filename to improve language recognition in lizard
        ext_map = {
            "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java",
            "c++": ".cpp", "c": ".c", "go": ".go", "ruby": ".rb", "c#": ".cs", "php": ".php",
            "rust": ".rs", "kotlin": ".kt", "swift": ".swift"
        }
        pseudo = "code" + ext_map.get(lang_key, ".txt")
        try:
            result = lizard.analyze_file.analyze_source_code(pseudo, code)
            fn_ccns = [fn.cyclomatic_complexity for fn in result.function_list] or [0]
            av = round(sum(fn_ccns) / max(len(fn_ccns), 1), 2)
            nloc = getattr(result, "nloc", None)
            long_lines = sum(1 for line in code.splitlines() if len(line) > 100)
            return {
                "avg_cyclomatic_complexity": av,
                "functions": len(result.function_list),
                "nloc": nloc,
                "long_lines_over_100": long_lines,
            }
        except Exception:
            pass
    # fallback minimal metrics
    return {
        "avg_cyclomatic_complexity": None,
        "functions": None,
        "nloc": len(code.splitlines()),
        "long_lines_over_100": sum(1 for line in code.splitlines() if len(line) > 100),
    }

def quick_autofix(code: str, comment: str, lang_key: str) -> str:
    c = comment.lower()
    if lang_key == "python":
        import autopep8
        fixed = autopep8.fix_code(code, options={"aggressive": 1})
        if "== true" in c or "== false" in c or "boolean" in c or "redundant" in c:
            # very basic boolean simplification
            fixed = re.sub(r"(\b[^=\n]+?)\s*==\s*True\b", r"\1", fixed)
            fixed = re.sub(r"(\b[^=\n]+?)\s*==\s*False\b", r"not (\1)", fixed)
        return fixed
    if lang_key in ["javascript", "typescript"]:
        out = code
        if "==" in code and "===" not in code:
            out = re.sub(r"(?<![=!])==(?![=])", "===", out)
        out = re.sub(r"\bvar\s+", "let ", out)
        return out
    if lang_key in ["c++", "c"]:
        out = code
        out = re.sub(r"\bNULL\b", "nullptr", out)  # C++
        return out
    if lang_key == "java":
        out = code
        out = re.sub(r"\bSystem\.out\.println\(", "System.out.println(", out)  # noop but placeholder
        return out
    if lang_key == "go":
        return code  # rely on LLM notes; gofmt not available here
    if lang_key == "ruby":
        out = code
        out = re.sub(r"\bTrue\b", "true", out)
        out = re.sub(r"\bFalse\b", "false", out)
        return out
    return code

def compile_ok_python(py_code: str) -> Tuple[bool, Optional[str]]:
    try:
        compile(py_code, "<suggested>", "exec")
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------- LLM Client By Vasu Johri----------------------------
SYSTEM_MENTOR_BASE = """You are an empathetic, precise senior engineer and mentor.
You give warm, actionable feedback with correct technical reasoning.
You tailor suggestions to the given programming language and cite a one-line resource link.
Include a concise improved code snippet when helpful. Keep it brief and specific.
Avoid shaming language. Prefer concrete references (style guides, best practices)."""

PERSONA_TONES = {
    "Supportive": "Use encouraging, friendly phrasing; assume a junior engineer.",
    "Direct": "Be concise and straightforward but still respectful.",
    "Senior-to-Senior": "Assume an experienced audience; focus on impact, accuracy, and performance.",
    "Non-native-friendly": "Use simple vocabulary and short sentences."
}

class LLMClient:
    def __init__(self, model: Optional[str] = None, persona: str = "Supportive", api_key: Optional[str] = None):
        self.model = model or os.environ.get("MODEL_NAME", "gpt-4.1")
        self.enabled = _OPENAI_AVAILABLE and bool(api_key or os.environ.get("OPENAI_API_KEY"))
        self.persona = persona
        if self.enabled:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = key
            self.client = OpenAI()
        else:
            self.client = None

    def chat(self, system: str, user: str, max_tokens: int = 900, temperature: float = 0.3) -> str:
        if not self.enabled:
            # Heuristic stand-in: trim user and prepend persona note
            user = re.sub(r"\s+", " ", user).strip()
            prefix = f"[Heuristic {self.persona} mentor] "
            return prefix + user[:1500]
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[{"role":"system","content":system},{"role":"user","content":user}],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            return getattr(resp, "output_text", str(resp))
        except Exception as e:
            return f"[LLM error: {e}]"


# ---------------------------- Reviewer Core Logic Building----------------------------
@dataclass
class ReviewInput:
    code_snippet: str
    review_comments: List[str]
    persona: str = "Supportive"
    model: Optional[str] = None
    language: Optional[str] = None

class EmpatheticReviewer:
    def __init__(self, persona: str = "Supportive", model: Optional[str] = None, api_key: Optional[str] = None, language: Optional[str] = None):
        self.persona = persona
        self.language = (language or "python").lower()
        self.llm = LLMClient(model=model, persona=persona, api_key=api_key)

    def _resource_for_language(self, lang_key: str) -> str:
        return LANG_RESOURCES.get(lang_key.lower(), "General best practices – https://12factor.net/")

    def _llm_section(self, code: str, comment: str, route: str, link: str, lang_key: str) -> str:
        system = SYSTEM_MENTOR_BASE + "\n" + PERSONA_TONES.get(self.persona, PERSONA_TONES["Supportive"])
        user = f"""Language: {lang_key}
Primary resource: {self._resource_for_language(lang_key)}
Context: The comment has been classified as **{route}**.

Code:
{code}

Original Comment:
"{comment}"

Write a short, structured review in Markdown with exactly these sections:
* **Positive Rephrasing:**
* **The 'Why':**
* **Suggested Improvement:** (include a concise {lang_key} code block if relevant)
* **Resource:** (a single, authoritative link)"""
        return self.llm.chat(system, user, max_tokens=900)

    def _heuristic_section(self, code: str, comment: str, route: str, lang_key: str) -> str:
        pos = f"Great start! A small tweak will improve {route.lower()} in {lang_key}."
        why = f"This aligns with common {lang_key} practices and helps readability and maintenance."
        improved = quick_autofix(code, comment, lang_key)
        fence = code_fence_lang(lang_key)
        section = "* **Positive Rephrasing:** " + pos + "\n" + \
                  "* **The 'Why':** " + why + "\n" + \
                  "* **Suggested Improvement:**\n" + \
                  f"```{fence}\n{improved}\n```\n" + \
                  "* **Resource:** " + self._resource_for_language(lang_key) + "\n"
        return section

    def _classify_comment(self, comment: str) -> Tuple[str, str]:
        c = comment.lower()
        # Minimal language-agnostic routing
        if any(k in c for k in ["name", "rename", "camel", "snake", "pascal"]):
            return "Naming", "Naming conventions / style"
        if any(k in c for k in ["== true", "== false", "boolean", "strict equality", "==="]):
            return "Style/Comparisons", "Boolean and comparison idioms"
        if any(k in c for k in ["read", "format", "style", "indent", "line length", "lint"]):
            return "Style/Readability", "Linting and formatting"
        if any(k in c for k in ["perf", "inefficient", "optimiz", "loop", "vectorize", "slow"]):
            return "Performance", "Algorithmic or idiomatic performance"
        if any(k in c for k in ["security", "unsafe", "injection", "eval(", "exec("]):
            return "Security", "Avoid dangerous constructs"
        if any(k in c for k in ["complex", "nested", "cyclomatic"]):
            return "Complexity", "Reduce branching depth"
        if any(k in c for k in ["doc", "comment", "explain"]):
            return "Docs/Comments", "Documentation quality"
        return "General", "General best practice"

    def analyze_one(self, code: str, comment: str, lang_key: str) -> Tuple[str, str, Dict]:
        route, _hint = self._classify_comment(comment)
        link = self._resource_for_language(lang_key)
        if self.llm.enabled:
            section = self._llm_section(code, comment, route, link, lang_key)
        else:
            section = self._heuristic_section(code, comment, route, lang_key)

        # Try to extract a code block
        fence = code_fence_lang(lang_key)
        m = re.search(rf"```(?:{fence}|{lang_key}|[a-zA-Z]+)?\s*([\s\S]*?)```", section)
        improved = m.group(1) if m else quick_autofix(code, comment, lang_key)

        ok, err = (True, None)
        if lang_key == "python":
            ok, err = compile_ok_python(improved)

        extra = {"route": route, "resource": link, "compile_ok": ok, "compile_error": err}
        return section, improved, extra

    def generate_markdown(self, data: ReviewInput) -> Tuple[str, str, Dict, Dict]:
        code = data.code_snippet
        lang_key = (data.language or self.language or "python").lower()
        metrics_before = metrics_for_any(code, lang_key)

        header = f"# Empathetic Code Review Report\n\n**Persona:** {self.persona}\n**Language:** {lang_key}\n\n"
        body = [header, "## Original Code\n", f"```{code_fence_lang(lang_key)}\n{code}\n```\n", "\n---\n", "## Feedback Analysis\n"]

        best_improved = code
        for idx, c in enumerate(data.review_comments, 1):
            section, improved, meta = self.analyze_one(code, c, lang_key)
            sec_title = f"### {idx}. Analysis of Comment: \"{c}\"  \n_Type: {meta['route']} • Resource: {meta['resource']} • Compile:_ {'✅ OK' if meta['compile_ok'] else '—'}"
            body.extend([sec_title, "\n\n", section, "\n---\n"])
            best_improved = improved  # last suggestion for diff

        metrics_after = metrics_for_any(best_improved, lang_key)
        diff = unified_diff(code, best_improved, fromfile=f"original.{code_fence_lang(lang_key)}", tofile=f"improved.{code_fence_lang(lang_key)}")

        # Metrics delta
        deltas = []
        for k in ["avg_cyclomatic_complexity", "functions", "nloc", "long_lines_over_100"]:
            b = metrics_before.get(k)
            n = metrics_after.get(k)
            if b is not None and n is not None:
                deltas.append(f"- **{k}**: {b} → {n}")
        delta_text = "\n".join(deltas) if deltas else "No metric deltas available."

        body.extend([
            "## Before/After Diff\n",
            f"```diff\n{diff}\n```\n",
            "\n## Metrics Delta\n",
            delta_text, "\n"
        ])

        # Summary
        body.extend(["\n## Summary\n", "Great progress! Focus on idiomatic style, clarity, and safe patterns to speed up reviews and reduce defects. 🙌"])

        # Footer credit
        body.extend(["\n---\n", "_Developed by **Vasu Johri**_"])

        report = "".join(body)
        return report, best_improved, metrics_before, metrics_after


# ---------------------------- Streamlit UI (For better User Experience) ----------------------------
st.set_page_config(page_title="Empathetic Code Reviewer — Multi‑Language Support", page_icon="🧠", layout="wide")
st.title("🧠 Empathetic Code Reviewer — Mission 1 (Multi‑Language Support)")

with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Mentor Persona", ["Supportive", "Direct", "Senior-to-Senior", "Non-native-friendly"], index=0)
    language_choice = st.selectbox("Target Language", LANG_LIST, index=0)
    use_llm = st.checkbox("Use OpenAI GPT (optional)", value=False)
    api_key = st.text_input("OpenAI API Key (optional)", type="password") if use_llm else None
    model = st.text_input("Model name", value="gpt-4.1") if use_llm else None
    st.markdown("---")
    st.caption("Tip: Works without GPT (heuristic mode). GPT improves pedagogy and examples.")

st.markdown("Paste your **code** on the left, **review comments** (one per line) on the right, then click **Generate Review**.")

col1, col2 = st.columns(2, gap="large")

with col1:
    code = st.text_area("Your Code", height=360, value="int add(int a, int b){\n  if(a==b) return a + b; else return a + b; // redundant branch\n}\n")

with col2:
    comments_raw = st.text_area("Reviewer Comments (one per line)", height=360, value="Use consistent braces and early returns.\nPrefer strict equality where relevant.\nReduce redundant branches; simplify logic.")

run_btn = st.button("Generate Review", type="primary", use_container_width=True)

# Optional: JSON upload (must contain code_snippet + review_comments + (optional) language)
uploaded = st.file_uploader("Or upload JSON with { code_snippet, review_comments, language? }", type=["json"])
if uploaded is not None:
    try:
        import json
        data_j = json.load(uploaded)
        code = data_j.get("code_snippet", code)
        rc = data_j.get("review_comments", [])
        if rc:
            comments_raw = "\n".join(rc)
        lg = data_j.get("language", None)
        if lg:
            language_choice = lg
        st.success("Loaded JSON successfully.")
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")

if run_btn:
    comments = [c.strip() for c in comments_raw.splitlines() if c.strip()]
    if not code.strip():
        st.error("Please paste your code.")
        st.stop()
    if not comments:
        st.warning("No comments provided; using a default readability note.")
        comments = ["Consider improving readability in the main function."]

    # Language detection/selection
    lang_key = guess_language(code, language_choice)

    reviewer = EmpatheticReviewer(persona=persona, model=(model if use_llm else None), api_key=(api_key if use_llm else None), language=lang_key)
    data = ReviewInput(code_snippet=code, review_comments=comments, persona=persona, model=(model if use_llm else None), language=lang_key)
    report, improved, metrics_before, metrics_after = reviewer.generate_markdown(data)

    st.subheader("Report (Markdown)")
    st.markdown(report, unsafe_allow_html=False)

    st.subheader("Improved Code (from suggestion)")
    st.code(improved, language=FENCE_ALIASES.get(lang_key, "text"))

    # Offer downloads
    st.download_button("⬇️ Download Report (.md)", data=report.encode("utf-8"), file_name=f"{slugify('empathetic_report')}.md", mime="text/markdown")
    st.download_button("⬇️ Download Improved Code", data=improved.encode("utf-8"), file_name=f"improved.{FENCE_ALIASES.get(lang_key, 'txt')}", mime="text/plain")

    with st.expander("Metrics (Before → After)"):
        st.json({"before": metrics_before, "after": metrics_after})

st.caption("© Mission 1 — Empathetic Code Reviewer (Multi‑Language Support). Works offline; GPT optional for premium guidance.  \nDeveloped by **Vasu Johri**")
