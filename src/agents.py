import litellm
import json
from .schema import GlobalState
from .prompts import SYSTEM_PROMPT_EXTRACTOR

def extract_state(history: list, current_state: GlobalState) -> GlobalState:
    """Uses LLM to analyze history and update the State Map."""
    
    # Convert Pydantic schema to a string so the AI knows the format
    schema_desc = GlobalState.model_json_schema()
    
    response = litellm.completion(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTOR.format(schema_json=schema_desc)},
            {"role": "user", "content": f"Current State: {current_state.json()}\n\nChat History: {history}"}
        ],
        response_format={ "type": "json_object" } # Forces JSON output
    )
    
    new_data = json.loads(response.choices[0].message.content)
    return GlobalState(**new_data)