# Demo cache

Cached LLM outputs used in DEMO MODE (when `ANTHROPIC_API_KEY` is not set).
The file name is `<prompt_name>.json`, or `<prompt_name>__<demo_key>.json` for multiple cases.
Each file must validate against the step's Pydantic output model. Tests check this.
