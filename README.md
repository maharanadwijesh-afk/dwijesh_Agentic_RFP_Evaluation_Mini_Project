# Agentic RFP Evaluation Mini Project (LLM-enabled)

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the sidebar password field. The default model is `gpt-4o-mini`; change it to a model enabled for your API account. API usage may incur charges. The key is not stored in SQLite or included in exported JSON.

## Workflow
1. Extract text from uploaded PDFs with pypdf.
2. Send extracted text and active criteria to the LLM.
3. Request JSON criterion scores, justifications, evidence, risks, and summary.
4. Validate scores; missing criterion results are set to zero and out-of-range values are clipped.
5. Python calculates weighted scores, benchmarks, gaps, relative percentages, PPI, and deterministic rankings.
6. Save results to SQLite and provide JSON export.

## Scoring and ranking
- Absolute score = sum((score / max score) × weight).
- Criterion benchmark = highest score for that criterion across suppliers.
- Gap = supplier score − benchmark.
- Relative performance = score / benchmark × 100 (zero if benchmark is zero).
- PPI = weighted average of relative-performance percentages.
- Tie-breaks: higher PPI, earlier submission date, higher experience rating, supplier name ascending.

## Notes
The LLM is instructed to treat document text as untrusted and to score only evidence present in the proposal. Scanned PDFs need OCR, which is not included. Extracted text is limited to 60,000 characters per proposal. Protect public deployments with authentication and usage limits. The five supplier PDFs are fictional training data.
