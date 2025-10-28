"""
AI Troubleshooter Agent - Day 1 Prototype

Features:
- Accepts pasted production incident content (logs, description, questions)
- Uses an LLM (LangChain + ChatOpenAI) to analyze the content and return structured
  thoughts: hypotheses, evidence, suggested next actions.
- Enforces structured JSON output using Pydantic models and LangChain's PydanticOutputParser.

Design notes (from our previous conversation):
- Evidence-first: every hypothesis should include evidence snippets (if present in the input).
- Structured output (JSON) to enable programmatic integration later.
- Low-temperature LLM for deterministic outputs.
- Read-only analysis only (no destructive actions in this prototype).

How to run:
1) Install dependencies:
   pip install langchain openai pydantic python-dotenv

2) Set OPENAI_API_KEY in your environment (or create a .env file in the same dir):
   export OPENAI_API_KEY="sk-..."

3) Run with:
   python ai_troubleshooter_day1.py --input-file sample_incident.txt
   OR
   python ai_troubleshooter_day1.py  # then paste input, end with EOF (Ctrl-D)

"""

from typing import List, Optional
from pydantic import BaseModel, Field
import re
import argparse
import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Load .env if present
load_dotenv()

# --------------------
# Output schema (Pydantic)
# --------------------
class EvidenceItem(BaseModel):
    source: Optional[str] = Field(None, description="Where this evidence came from (e.g., 'input-line-23' or 'user-description')")
    text: str = Field(..., description="Exact evidence line or snippet")

class Hypothesis(BaseModel):
    id: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ActionItem(BaseModel):
    id: str
    description: str
    command: Optional[str] = None
    expected_result: Optional[str] = None
    requires_approval: bool = False
    risk: Optional[str] = None

class AgentOutput(BaseModel):
    summary: str
    hypotheses: List[Hypothesis]
    suggested_actions: List[ActionItem]
    notes: Optional[str] = None

# --------------------
# Simple extraction helpers (to give the LLM some structured context)
# --------------------

ERROR_REGEX = re.compile(r"(Exception|Error|Timeout|401|403|500|502|503|504|timeout|failed|cannot|unable)", re.IGNORECASE)
EXCEPTION_LINE_REGEX = re.compile(r"([A-Za-z0-9_.]+Exception)[:\s-]*(.*)")
HTTP_CODE_REGEX = re.compile(r"\b(401|403|404|500|502|503|504)\b")
SERVICE_REGEX = re.compile(r"(AuthService|authservice|Redis|redis|Mobile App|mobile app|backend|frontend|database|DB|SessionStore)", re.IGNORECASE)


def extract_candidate_signals(text: str) -> dict:
    """Extract naive signals: exception names, http codes, service names, interesting lines."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    exceptions = []
    http_codes = set()
    services = set()
    interesting_lines = []

    for i, line in enumerate(lines):
        if ERROR_REGEX.search(line):
            interesting_lines.append((i + 1, line))
        m = EXCEPTION_LINE_REGEX.search(line)
        if m:
            exceptions.append(m.group(1))
            interesting_lines.append((i + 1, line))
        for code_m in HTTP_CODE_REGEX.findall(line):
            http_codes.add(code_m)
            interesting_lines.append((i + 1, line))
        for s in SERVICE_REGEX.findall(line):
            services.add(s)

    return {
        "exceptions": list(dict.fromkeys(exceptions)),
        "http_codes": sorted(list(http_codes)),
        "services": sorted(list(services)),
        "interesting_lines": interesting_lines,
        "raw_lines": lines,
    }

# --------------------
# Prompt template & LLM setup
# --------------------

SYSTEM_PROMPT = """
You are an evidence-first incident troubleshooting assistant. 
You will be given raw user input containing incident logs, descriptions, and questions.
Produce a structured JSON object that conforms exactly to the provided Pydantic schema for AgentOutput.

Rules (must follow):
1) Base your hypotheses on the evidence present in the provided input lines whenever possible. 
   For each hypothesis include evidence list entries indicating the source (e.g., 'input-line-12') and the exact snippet.
2) Do NOT hallucinate facts that are not in the input. If you have to speculate, mark the confidence lower and put your speculation in notes (or include 0.0-0.3 confidence).
3) Provide up to 4 hypotheses, ranked by confidence (0.0 - 1.0).
4) Provide up to 6 suggested actions. Actions should be non-destructive checks or read-only queries whenever possible.
5) When suggesting commands, prefer commonly used diagnostic commands (kubectl, curl, redis-cli, logs queries). Mark requires_approval True for write/destructive operations.
6) Keep the summary concise (2-4 sentences).
7) The output must be valid JSON matching the schema - do not output extra commentary outside the JSON.
"""

HUMAN_PROMPT = """
User pasted the following incident content between triple backticks:

"""
{user_input}
"""

Additionally parsed signals (naive extraction):
Exceptions: {exceptions}
HTTP codes: {http_codes}
Services found: {services}
Number of non-empty input lines: {num_lines}

Task:
1) Read the content and the parsed signals. Create a structured JSON of AgentOutput containing:
   - summary
   - hypotheses (each must include id, description, confidence, and evidence pointing to input lines)
   - suggested_actions (id, description, command (if any), expected result, requires_approval, risk)
   - optional notes

Remember: cite input lines as evidence using the format 'input-line-N' where N is the 1-based line number.
If there are no exact supporting lines for a hypothesis, the hypothesis can still be suggested but must have low confidence and the evidence array can be empty.
"""

# --------------------
# Core agent function
# --------------------

def build_agent(llm_model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Construct the LLM and the prompt parser.

    Returns (llm, prompt, parser)
    """
    # Chat model - low temperature for deterministic output
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    # Create a PydanticOutputParser from our schema
    parser = PydanticOutputParser(pydantic_object=AgentOutput)

    system_msg = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

    # We will plug the user_input and extracted signals into the human prompt
    human_template = HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)

    prompt = ChatPromptTemplate.from_messages([system_msg, human_template])

    return llm, prompt, parser


def analyze_input(user_input: str, llm, prompt, parser) -> AgentOutput:
    """Run the LLM with extracted signals and parse the structured result."""
    signals = extract_candidate_signals(user_input)

    # Prepare the prompt input
    prompt_input = {
        "user_input": user_input,
        "exceptions": ", ".join(signals.get("exceptions") or []) or "(none found)",
        "http_codes": ", ".join(signals.get("http_codes") or []) or "(none found)",
        "services": ", ".join(signals.get("services") or []) or "(none found)",
        "num_lines": len(signals.get("raw_lines"))
    }

    # Format the messages
    messages = prompt.format_messages(**prompt_input)

    # Call the model
    llm_response = llm(messages)
    raw_output = llm_response.content

    # Parse using PydanticOutputParser - this will enforce schema and raise helpful errors
    parsed = parser.parse(raw_output)

    return parsed

# --------------------
# CLI / simple run
# --------------------

def main():
    parser_arg = argparse.ArgumentParser(description="Day-1 AI Troubleshooter Agent (CLI)")
    parser_arg.add_argument("--input-file", type=str, help="Path to file containing incident content (optional)")
    parser_arg.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="LLM model name to use")
    parser_arg.add_argument("--temp", type=float, default=0.0, help="LLM temperature (0.0–1.0)")

    args = parser_arg.parse_args()

    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Input file {args.input_file} not found")
            return
        with open(args.input_file, "r", encoding="utf-8") as f:
            user_input = f.read()
    else:
        print("Paste your incident content now (end with EOF / Ctrl-D):")
        user_input = "\n".join(iter(input, ""))

    # Build agent
    llm, prompt, out_parser = build_agent(llm_model_name=args.model, temperature=args.temp)

    try:
        result = analyze_input(user_input, llm, prompt, out_parser)
    except Exception as e:
        print("Error during analysis:", e)
        print("\nAttempting to print raw LLM output for debugging (if available) — note: this may be unstructured):\n")
        # Try to call the model once more to get raw text
        try:
            messages = prompt.format_messages(**{
                "user_input": user_input,
                "exceptions": "(error)",
                "http_codes": "(error)",
                "services": "(error)",
                "num_lines": len(user_input.splitlines())
            })
            raw = llm(messages).content
            print(raw)
        except Exception:
            print("Could not fetch raw LLM output")
        return

    # Print pretty JSON
    out_json = result.json(indent=2, ensure_ascii=False)
    print(out_json)


if __name__ == "__main__":
    main()
