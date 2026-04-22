"""
SSM Framework - Agent Logic Module
==================================

This module implements the core agentic workflow using LangGraph for
orchestration. It defines the nodes and edges of the reasoning graph
that powers the SSM framework's state management capabilities.

Features:
- Gemini 2.5 Flash as the primary LLM
- langsmith integration for tracing and monitoring
- 8-node agent workflow for state management
- Conflict detection and resolution

The agent workflow follows this lifecycle:
1. Input Capture - Receive and validate user input
2. Extraction - Parse intent into structured state using LLM
3. Conflict Detection - Compare extraction against current state
4. State Synchronization - Update state map if no conflicts
5. Prompt Augmentation - Inject state anchors into prompts
6. Generation - Get response from target LLM
7. Reflexion - Audit response for constraint adherence
8. Response Delivery - Return final response to user
"""

import os
import json
import re
import uuid
import langsmith
from typing import TypedDict, Literal, Optional, Dict, Any, List
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END

# langsmith imports for tracing
from langsmith import Client
from langsmith.run_helpers import traceable

# Pydantic for validation
from pydantic import BaseModel, ValidationError

# Load environment
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import (
    GlobalStateMap,
    ExtractionResult,
    ConflictRecord,
    LanguageEnum,
    ParadigmEnum
)
from src.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_augmented_prompt,
    build_clarification_prompt,
    build_reflexion_prompt,
    format_state_summary
)
from src.memory import MemoryManager


# =============================================================================
# langsmith CONFIGURATION
# =============================================================================

class langsmithConfig:
    """Configuration for langsmith tracing and monitoring."""
    
    # langsmith settings from environment
    API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    PROJECT = os.getenv("LANGSMITH_PROJECT", "SSM-Framework")
    
    # Initialize langsmith client if configured
    client: Optional[Client] = None
    
    @classmethod
    def initialize(cls) -> bool:
        """Initialize langsmith client."""
        if not cls.API_KEY:
            print(" langsmith API key not configured. Tracing disabled.")
            return False
        
        try:
            cls.client = Client(
                api_url="https://api.smith.langchain.com"
            )
            print(f" langsmith initialized. Project: {cls.PROJECT}")
            return True
        except Exception as e:
            print(f" Failed to initialize langsmith: {e}")
            return False
    
    @classmethod
    def create_run(
        cls,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Create a new langsmith run and return its ID."""
        if not cls.client:
            return None
        
        try:
            run = cls.client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                project_name=cls.PROJECT,
                tags=tags or []
            )
            return run.id if run else None
        except Exception as e:
            print(f"langsmith run creation error: {e}")
            return None
    
    @classmethod
    def update_run(
        cls,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Update a langsmith run with outputs or error."""
        if not cls.client or not run_id:
            return
        
        try:
            cls.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error
            )
        except Exception as e:
            print(f"langsmith update error: {e}")
    
    @classmethod
    def get_run_url(cls, run_id: str) -> Optional[str]:
        """Get the URL to view a run in langsmith."""
        if not run_id:
            return None
        return f"https://smith.langchain.com/o/default/projects/p/{cls.PROJECT}/runs/{run_id}"


langsmith = langsmithConfig
LangSmithConfig = langsmithConfig


# =============================================================================
# LLM CONFIGURATION - GEMINI 2.5 FLASH
# =============================================================================

class LLMConfig:
    """
    Configuration for LLM interactions using Gemini 2.5 Flash.
    
    All model settings are configurable via environment variables.
    """
    
    # Model configuration - Gemini 2.5 Flash
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
    EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini-2.5-flash")
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
    
    # Generation parameters
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
    
    # Gemini API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are configured."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        return True


# =============================================================================
# GEMINI LLM CLIENT
# =============================================================================

class GeminiClient:
    """
    Client for interacting with Gemini 2.5 Flash via the Gemini API.
    
    Provides a unified interface for LLM calls with langsmith tracing.
    """
    
    def __init__(self):
        """Initialize the Gemini client."""
        LLMConfig.validate()
        
        # Import Google's generative AI library
        try:
            import google.generativeai as genai
            genai.configure(api_key=LLMConfig.GEMINI_API_KEY)
            self.genai = genai
            self.client_initialized = True
            print(" Gemini 2.5 Flash client initialized")
        except ImportError:
            print(" google-generativeai not installed. Install with: pip install google-generativeai")
            self.client_initialized = False
        except Exception as e:
            print(f" Failed to initialize Gemini client: {e}")
            self.client_initialized = False
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            model: Model to use (defaults to configured model)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            run_name: Name for langsmith tracing
            
        Returns:
            The generated text response
        """
        if not self.client_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        # Use defaults from config
        model_name = model or LLMConfig.GENERATION_MODEL
        temp = temperature if temperature is not None else LLMConfig.TEMPERATURE
        max_out = max_tokens or LLMConfig.MAX_TOKENS
        
        # Create langsmith run for tracing
        run_id = None
        if langsmith.client and run_name:
            run_id = langsmith.create_run(
                name=run_name,
                run_type="llm",
                inputs={
                    "prompt": prompt[:500],  # Truncate for logging
                    "model": model_name,
                    "temperature": temp
                },
                tags=["gemini", "ssm-framework"]
            )
        
        try:
            # Configure the model
            generation_config = self.genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_out,
                response_mime_type="text/plain"
            )
            
            # Create the model instance
            model_instance = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            # Generate response
            response = model_instance.generate_content(prompt)
            
            result_text = response.text
            
            # Update langsmith run with success
            if run_id:
                langsmith.update_run(
                    run_id=run_id,
                    outputs={"response": result_text[:500]}
                )
            
            return result_text
            
        except Exception as e:
            error_msg = f"Gemini generation error: {str(e)}"
            
            # Update langsmith run with error
            if run_id:
                langsmith.update_run(run_id=run_id, error=error_msg)
            
            raise RuntimeError(error_msg)
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from Gemini.
        
        Uses controlled generation to ensure valid JSON output.
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            model: Model to use
            temperature: Temperature for generation
            run_name: Name for langsmith tracing
            
        Returns:
            Parsed JSON dictionary
        """
        if not self.client_initialized:
            raise RuntimeError("Gemini client not initialized")
        
        model_name = model or LLMConfig.EXTRACTION_MODEL
        temp = temperature if temperature is not None else 0.3  # Lower temp for JSON
        
        # Create langsmith run for tracing
        run_id = None
        if langsmith.client and run_name:
            run_id = langsmith.create_run(
                name=run_name,
                run_type="llm",
                inputs={"prompt": prompt[:500], "model": model_name},
                tags=["gemini", "json", "ssm-framework"]
            )
        
        try:
            # Configure for JSON output
            generation_config = self.genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
            
            # Create the model instance
            model_instance = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            # Generate response
            response = model_instance.generate_content(prompt)
            
            # Parse JSON
            result_text = response.text
            result_json = json.loads(result_text)
            
            # Update langsmith run with success
            if run_id:
                langsmith.update_run(
                    run_id=run_id,
                    outputs={"response": result_text[:500]}
                )
            
            return result_json
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            if run_id:
                langsmith.update_run(run_id=run_id, error=error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Gemini JSON generation error: {str(e)}"
            if run_id:
                langsmith.update_run(run_id=run_id, error=error_msg)
            raise RuntimeError(error_msg)


# =============================================================================
# GRAPH STATE TYPE
# =============================================================================

class GraphState(TypedDict):
    """State type for the LangGraph workflow."""
    user_input: str
    global_state: Optional[GlobalStateMap]
    extraction_result: Optional[ExtractionResult]
    detected_conflicts: List[ConflictRecord]
    augmented_prompt: Optional[str]
    llm_response: Optional[str]
    final_response: Optional[str]
    needs_clarification: bool
    force_merge: bool
    merged_fields: List[str]
    error: Optional[str]
    memory_manager: Optional[MemoryManager]
    langsmith_run_id: Optional[str]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_language(lang_str: Optional[str]) -> Optional[str]:
    """
    Normalize language string to match LanguageEnum values.
    
    Maps various formats (C++, Python, JAVA, etc.) to canonical lowercase names.
    This ensures consistent language identification regardless of user input format.
    
    Args:
        lang_str: Raw language string from LLM or user
        
    Returns:
        Normalized language value matching LanguageEnum, or None if not recognized
    """
    if not lang_str:
        return None
    
    # Normalize: strip whitespace, lowercase for matching
    normalized = lang_str.strip().lower()
    
    # Direct mapping of common language names and variants
    language_map = {
        # Python
        "python": "python",
        "py": "python",
        "python3": "python",
        "python 3": "python",
        "python3.11": "python",
        "python3.10": "python",
        
        # JavaScript
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "node.js": "javascript",
        "nodejs": "javascript",
        
        # TypeScript
        "typescript": "typescript",
        "ts": "typescript",
        
        # Java
        "java": "java",
        "jdk": "java",
        
        # C#
        "c#": "csharp",
        "c-sharp": "csharp",
        "csharp": "csharp",
        "cs": "csharp",
        ".net": "csharp",
        "dotnet": "csharp",
        ".net core": "csharp",
        ".net 8": "csharp",
        
        # C++
        "c++": "cpp",
        "cpp": "cpp",
        "c plus plus": "cpp",
        "cxx": "cpp",
        "c plus": "cpp",
        
        # Go
        "go": "go",
        "golang": "go",
        
        # Rust
        "rust": "rust",
        "rs": "rust",
        
        # PHP
        "php": "php",
        
        # Kotlin
        "kotlin": "kotlin",
        "kt": "kotlin",
        
        # Swift
        "swift": "swift",
        
        # Ruby
        "ruby": "ruby",
        "rb": "ruby",
        "rails": "ruby",
        "ruby on rails": "ruby",
        
        # SQL
        "sql": "sql",
        "plsql": "sql",
        "t-sql": "sql",
        "tsql": "sql",
        "postgresql": "sql",
        "postgres": "sql",
        "mysql": "sql",
        "oracle": "sql",
        "sqlite": "sql",
    }
    
    # Exact match first
    if normalized in language_map:
        return language_map[normalized]
    
    # Try prefix matching for compound names
    for key, value in language_map.items():
        if normalized.startswith(key):
            return value
    
    # Return None if not recognized (will default to "other")
    return None


def has_explicit_language_request(user_input: str) -> bool:
    """
    Detect whether the user explicitly named a programming language.

    This helps preserve the current language for generic code requests
    instead of letting nearby context or the extractor flip the state.
    """
    if not user_input:
        return False

    text = user_input.lower()
    markers = [
        "python",
        "java",
        "javascript",
        "typescript",
        "c#",
        "csharp",
        "c++",
        "cpp",
        "go",
        "rust",
        "kotlin",
        "swift",
        "ruby",
        "php",
        "sql",
    ]

    return any(marker in text for marker in markers)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def node_input_capture(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture and validate user input.
    
    Entry point for the agent workflow. Validates that we have
    meaningful input to process and initializes resources.
    """
    user_input = state.get("user_input", "").strip()
    
    if not user_input:
        return {
            "error": "Empty input received",
            "final_response": "I didn't receive any input. How can I help you?"
        }
    
    # Initialize memory manager if not present
    if "memory_manager" not in state or state["memory_manager"] is None:
        state["memory_manager"] = MemoryManager()
    
    # Create langsmith run for this turn
    run_id = None
    if langsmith.client:
        run_id = langsmith.create_run(
            name=f"ssm_turn_{datetime.now().strftime('%H%M%S')}",
            run_type="chain",
            inputs={"user_input": user_input},
            tags=["ssm-framework", "conversation-turn"]
        )
    
    return {
        "user_input": user_input,
        "error": None,
        "force_merge": state.get("force_merge", False),
        "merged_fields": state.get("merged_fields", []),
        "langsmith_run_id": run_id
    }


def node_extraction(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured state from user input.
    
    Uses Gemini 2.5 Flash to parse the user's intent and extract
    any relevant state attributes into structured format.
    """
    if state.get("error"):
        return state
    
    user_input = state["user_input"]
    memory_manager = state.get("memory_manager")
    global_state = memory_manager.state_map if memory_manager else GlobalStateMap()
    current_language = global_state.tech.language.value if global_state.tech.language else None
    explicit_language_request = has_explicit_language_request(user_input)
    
    # Get relevant context from episodic memory
    relevant_context = None
    if memory_manager:
        relevant_context = memory_manager.find_relevant_context(user_input)
    
    # Build extraction prompt
    extraction_prompt = build_extraction_prompt(
        user_input=user_input,
        current_state=global_state,
        relevant_history=relevant_context
    )
    
    try:
        # Initialize Gemini client
        gemini = GeminiClient()
        
        # Call Gemini for extraction with JSON output
        extraction_result = gemini.generate_json(
            prompt=extraction_prompt,
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            model=LLMConfig.EXTRACTION_MODEL,
            run_name="state_extraction"
        )
        
        # Validate and create ExtractionResult
        if extraction_result:
            extraction = ExtractionResult(**extraction_result)
            
            # Normalize language if extracted
            if extraction.extracted_state and "tech" in extraction.extracted_state:
                tech_state = extraction.extracted_state["tech"]
                if "language" in tech_state:
                    raw_lang = tech_state["language"]
                    print(f"[DEBUG] Raw language from Gemini: {repr(raw_lang)}")
                    
                    # Normalize the language
                    normalized_lang = normalize_language(raw_lang)
                    
                    if normalized_lang:
                        if current_language and normalized_lang != current_language and not explicit_language_request:
                            tech_state["language"] = current_language
                            print(f"[DEBUG] Preserving current language: {current_language}")
                        else:
                            tech_state["language"] = normalized_lang
                            print(f"[DEBUG] Normalized language: {normalized_lang}")
                    else:
                        if current_language and not explicit_language_request:
                            tech_state["language"] = current_language
                            print(f"[DEBUG] Preserving current language: {current_language}")
                        else:
                            # If normalization fails, set to "other"
                            tech_state["language"] = "other"
                            print(f"[DEBUG] Language not recognized, defaulting to 'other'")
                else:
                    if current_language and not explicit_language_request:
                        tech_state["language"] = current_language
                        print(f"[DEBUG] No language field in extraction, preserving current language: {current_language}")
                    else:
                        # Language field missing - set to "other"
                        tech_state["language"] = "other"
                        print(f"[DEBUG] No language field in extraction, defaulting to 'other'")
            else:
                # No tech state - ensure it exists with language set
                if not extraction.extracted_state:
                    extraction.extracted_state = {}
                if current_language and not explicit_language_request:
                    extraction.extracted_state["tech"] = {"language": current_language}
                    print(f"[DEBUG] No tech state in extraction, preserving current language: {current_language}")
                else:
                    extraction.extracted_state["tech"] = {"language": "other"}
                    print(f"[DEBUG] No tech state in extraction, creating with 'other' language")
            
            # Ensure chain_of_thought is captured
            if 'chain_of_thought' in extraction_result and not extraction.chain_of_thought:
                extraction.chain_of_thought = extraction_result['chain_of_thought']
        else:
            extraction = ExtractionResult(
                raw_intent=user_input,
                confidence=0.3,
                extracted_state={"tech": {"language": "other"}}
            )
            print(f"[DEBUG] Gemini returned empty extraction, defaulting to 'other' language")
        
        return {
            "extraction_result": extraction,
            "detected_conflicts": list(extraction.detected_conflicts),
            "chain_of_thought": extraction.chain_of_thought
        }
        
    except Exception as e:
        # On extraction failure, proceed with low confidence
        print(f"Extraction error: {e}")
        return {
            "extraction_result": ExtractionResult(
                raw_intent=user_input,
                confidence=0.2
            ),
            "error": f"Extraction warning: {str(e)}"
        }


def node_conflict_detection(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect CRITICAL conflicts between extracted state and current state.
    
    ONLY reports conflicts for ACTUAL language changes where:
    - BOTH current AND new languages are explicitly set (not "other")
    - User is trying to switch from one language to another
    
    If force_merge=True, skip conflict reporting (user already accepted)
    """
    # If user already accepted changes, skip conflict detection
    if state.get("force_merge", False):
        return {
            **state,  # Preserve all state
            "detected_conflicts": [],
            "needs_clarification": False
        }
    
    if state.get("error") and "Extraction warning" not in state.get("error", ""):
        return state
    
    extraction = state.get("extraction_result")
    memory_manager = state.get("memory_manager")
    global_state = memory_manager.state_map if memory_manager else GlobalStateMap()
    
    if not extraction or extraction.confidence < 0.3:
        return state
    
    conflicts = []
    extracted = extraction.extracted_state
    
    # DEBUG: Log extraction and state comparison
    if "tech" in extracted and "language" in extracted["tech"]:
        new_lang = extracted["tech"]["language"]
        current_lang = global_state.tech.language.value if global_state.tech.language else None
        print(f"[CONFLICT CHECK] Current language: {current_lang} | Extracted language: {new_lang}")
    
    # ONLY check for REAL language conflicts
    if "tech" in extracted and "language" in extracted["tech"]:
        new_lang = extracted["tech"]["language"]
        current_lang = global_state.tech.language.value if global_state.tech.language else None
        
        # Only report conflict if ALL these are true:
        # 1. Languages are different
        # 2. Current language is explicitly set (not None and not "other")
        # 3. New language is explicitly detected (not None and not "other")
        
        # This means:
        # - NO conflict if current=None (truly uninitialized)
        # - NO conflict if current="other" (uninitialized legacy state)
        # - NO conflict if new="other" (couldn't detect language from input)
        # - ONLY conflict if current="python" and new="java" (real switch)
        
        is_current_initialized = current_lang and current_lang != "other"
        is_new_explicit = new_lang and new_lang != "other"
        is_different = new_lang != current_lang
        
        print(f"[CONFLICT EVAL] is_current_initialized={is_current_initialized}, is_new_explicit={is_new_explicit}, is_different={is_different}")
        
        if is_different and is_current_initialized and is_new_explicit:
            # This is a REAL conflict: switching from python → java (for example)
            print(f"[CONFLICT DETECTED] Real language change: {current_lang} → {new_lang}")
            conflicts.append(ConflictRecord(
                conflict_id=f"lang_{uuid.uuid4().hex[:8]}",
                field_path="tech.language",
                existing_value=current_lang,
                proposed_value=new_lang,
                severity="critical",
                status="detected"
            ))
        else:
            print(f"[NO CONFLICT] Silently merging language change")
    
    # Determine if clarification is needed
    # Only critical conflicts require user clarification
    needs_clarification = any(
        c.severity == "critical" for c in conflicts
    )
    
    return {
        **state,  # Preserve all state
        "detected_conflicts": conflicts,
        "needs_clarification": needs_clarification
    }


def node_state_sync(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize state map with extraction results using confidence thresholds.
    
    Updates the global state map with the extracted attributes,
    using confidence scores to determine merge priority and conflict severity.
    
    If force_merge=True (user chose to accept conflicts), overrides conflict detection.
    """
    if state.get("needs_clarification") and not state.get("force_merge"):
        # Don't merge if clarification needed, unless user explicitly forced merge
        return state
    
    extraction = state.get("extraction_result")
    memory_manager = state.get("memory_manager")
    force_merge = state.get("force_merge", False)
    
    if not extraction or not memory_manager:
        return state
    
    global_state = memory_manager.state_map
    
    # Determine confidence threshold and force flag
    # If user accepted conflicts, use force=True to override
    merge_confidence_threshold = 0.65
    if force_merge:
        merge_confidence_threshold = 0.45  # Lower threshold when user accepts
    
    # Merge extraction into state
    merged_conflicts = global_state.merge_extraction(
        extraction=extraction,
        force=force_merge,
        confidence_threshold=merge_confidence_threshold
    )
    
    # Track which fields were actually merged
    merged_fields = []
    if extraction.extracted_state.get('tech'):
        for field in extraction.extracted_state['tech'].keys():
            if field in global_state.tech.__fields__:
                merged_fields.append(field)
    
    # Save updated state
    memory_manager.save_state()
    
    return {
        "global_state": global_state,
        "merged_fields": merged_fields,
        "merged_conflicts": merged_conflicts
    }


def node_prompt_augmentation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Augment prompt with state anchors.
    
    Constructs the final prompt that will be sent to Gemini,
    including the state anchor that ensures constraint adherence.
    """
    if state.get("needs_clarification"):
        # Build clarification prompt instead
        conflicts = state.get("detected_conflicts", [])
        user_input = state["user_input"]
        clarification = build_clarification_prompt(conflicts, user_input)
        return {
            "augmented_prompt": clarification,
            "final_response": clarification
        }
    
    user_input = state["user_input"]
    memory_manager = state.get("memory_manager")
    global_state = memory_manager.state_map if memory_manager else GlobalStateMap()
    
    # Get relevant context
    relevant_context = None
    if memory_manager:
        relevant_context = memory_manager.find_relevant_context(user_input)
    
    # Build augmented prompt
    augmented_prompt = build_augmented_prompt(
        user_input=user_input,
        state_map=global_state,
        relevant_context=relevant_context
    )
    
    return {
        "augmented_prompt": augmented_prompt
    }


def node_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate response using Gemini 2.5 Flash.
    
    Sends the augmented prompt to Gemini and receives the response.
    """
    if state.get("needs_clarification"):
        return state  # Clarification already built
    
    augmented_prompt = state.get("augmented_prompt")
    
    if not augmented_prompt:
        return {
            "error": "No augmented prompt to send"
        }
    
    print(f"\n[GENERATION] Sending augmented prompt to Gemini")
    print(f"[GENERATION] Augmented prompt length: {len(augmented_prompt)}")
    # Log the state anchor portion (should have Language: java)
    if "TECHNICAL CONSTRAINTS" in augmented_prompt:
        anchor_start = augmented_prompt.find("TECHNICAL CONSTRAINTS")
        anchor_end = augmented_prompt.find("===", anchor_start + 10) + 20
        state_anchor = augmented_prompt[anchor_start:anchor_end]
        print(f"[GENERATION] State anchor in prompt:\n{state_anchor}")
    
    try:
        gemini = GeminiClient()
        
        llm_response = gemini.generate(
            prompt=augmented_prompt,
            model=LLMConfig.GENERATION_MODEL,
            temperature=LLMConfig.TEMPERATURE,
            max_tokens=LLMConfig.MAX_TOKENS,
            run_name="generation"
        )
        
        print(f"[GENERATION] Gemini response received (first 100 chars): {llm_response[:100]}")
        
        return {
            "llm_response": llm_response
        }
        
    except Exception as e:
        return {
            "error": f"Generation error: {str(e)}",
            "llm_response": None
        }


def node_reflexion(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audit the LLM response for constraint adherence.
    
    Reviews the generated response against the state map to verify
    that all constraints were followed.
    """
    if state.get("needs_clarification"):
        return state
    
    llm_response = state.get("llm_response")
    
    if not llm_response:
        return state
    
    memory_manager = state.get("memory_manager")
    global_state = memory_manager.state_map if memory_manager else GlobalStateMap()
    user_input = state["user_input"]
    
    # Build reflexion prompt
    reflexion_prompt = build_reflexion_prompt(
        llm_response=llm_response,
        state_map=global_state,
        original_request=user_input
    )
    
    try:
        gemini = GeminiClient()
        
        reflexion_result = gemini.generate_json(
            prompt=reflexion_prompt,
            model=LLMConfig.EXTRACTION_MODEL,
            temperature=0.2,
            run_name="reflexion"
        )
        
        if reflexion_result and reflexion_result.get("violations_found"):
            violations = reflexion_result.get("violations", [])
            print(f"⚠️ Reflexion found violations: {violations}")
            corrected_response = llm_response
            if violations:
                first_violation = violations[0]
                corrected_response = first_violation.get("corrected_response", llm_response)
                print(f"[REFLEXION] Original response (first 100 chars): {llm_response[:100]}")
                print(f"[REFLEXION] Corrected response (first 100 chars): {corrected_response[:100]}")
            return {
                "llm_response": corrected_response,
                "final_response": corrected_response,
                "reflexion_violations": violations
            }
        
    except (ValueError, Exception) as e:
        print(f"[REFLEXION] Error during reflection (JSON parse or other): {type(e).__name__}: {str(e)[:100]}")
        print(f"[REFLEXION] Using original response without correction")
    
    print(f"[REFLEXION] No violations found, using original response")
    return {
        "llm_response": llm_response,
        "final_response": llm_response
    }


def node_response_delivery(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deliver the final response to the user.
    
    Records the conversation turn in memory and prepares the
    final response.
    """
    if state.get("needs_clarification"):
        return state
    
    final_response = state.get("llm_response", state.get("final_response", ""))
    
    print(f"[RESPONSE DELIVERY] Final response source:")
    print(f"  - llm_response exists: {state.get('llm_response') is not None}")
    print(f"  - final_response exists: {state.get('final_response') is not None}")
    print(f"  - Using response (first 100 chars): {final_response[:100]}")
    
    # Record turn in memory
    memory_manager = state.get("memory_manager")
    run_id = state.get("langsmith_run_id")
    
    if memory_manager:
        memory_manager.record_turn(
            user_input=state["user_input"],
            extraction=state.get("extraction_result"),
            conflicts=state.get("detected_conflicts", []),
            response=final_response,
            langsmith_run_id=run_id
        )
    
    # Update langsmith run with final result
    if run_id and langsmith.client:
        langsmith.update_run(
            run_id=run_id,
            outputs={"final_response": final_response[:500]}
        )
    
    return {
        "final_response": final_response
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def route_after_conflict(state: Dict[str, Any]) -> Literal["end_conflict", "sync"]:
    """
    Route after conflict detection.
    
    - If conflicts detected AND no force_merge: end here (let UI handle conflicts)
    - If no conflicts OR force_merge accepted: proceed to sync
    """
    has_conflicts = bool(state.get("detected_conflicts"))
    force_merge = state.get("force_merge", False)
    
    if has_conflicts and not force_merge:
        # End the graph - let UI show conflicts and wait for user choice
        return "end_conflict"
    
    # No conflicts, or user forced merge - proceed to sync
    return "sync"


def node_end_conflict(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    End node that returns when conflicts are detected but not force-merged.
    
    Allows the UI to show the conflicts and wait for user choice.
    When the user accepts, the agent will be called again with force_merge=True.
    """
    return {
        "final_response": "Conflicts detected - waiting for your choice",
        "error": None
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph agent workflow.
    
    Constructs the complete graph with all nodes and edges
    for the SSM agent workflow.
    """
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("input_capture", node_input_capture)
    workflow.add_node("extraction", node_extraction)
    workflow.add_node("conflict_detection", node_conflict_detection)
    workflow.add_node("end_conflict", node_end_conflict)
    workflow.add_node("state_sync", node_state_sync)
    workflow.add_node("prompt_augmentation", node_prompt_augmentation)
    workflow.add_node("generation", node_generation)
    workflow.add_node("reflexion", node_reflexion)
    workflow.add_node("response_delivery", node_response_delivery)
    
    # Set entry point
    workflow.set_entry_point("input_capture")
    
    # Add edges
    workflow.add_edge("input_capture", "extraction")
    workflow.add_edge("extraction", "conflict_detection")
    
    # Conditional edge after conflict detection
    workflow.add_conditional_edges(
        "conflict_detection",
        route_after_conflict,
        {
            "end_conflict": "end_conflict",  # Has conflicts - end and let UI handle
            "sync": "state_sync"  # No conflicts - proceed to sync
        }
    )
    
    workflow.add_edge("end_conflict", END)
    workflow.add_edge("state_sync", "prompt_augmentation")
    workflow.add_edge("prompt_augmentation", "generation")
    workflow.add_edge("generation", "reflexion")
    workflow.add_edge("reflexion", "response_delivery")
    workflow.add_edge("response_delivery", END)
    
    return workflow.compile()


# =============================================================================
# AGENT CLASS
# =============================================================================

class SSMAgent:
    """
    Main SSM Agent class providing a simple interface for interactions.
    
    This class wraps the LangGraph workflow and provides methods for
    processing user input and managing state.
    """
    
    def __init__(
        self,
        state_dir: str = "./data/state",
        chroma_dir: str = "./data/chroma_db"
    ):
        """
        Initialize the SSM Agent.
        
        Args:
            state_dir: Directory for state persistence
            chroma_dir: Directory for ChromaDB storage
        """
        # Initialize langsmith
        langsmith.initialize()
        
        # Initialize Gemini client
        self._gemini = GeminiClient()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            state_dir=state_dir,
            chroma_dir=chroma_dir
        )
        
        # Build the agent graph
        self.graph = build_agent_graph()
        
        print("🧠 SSM Agent initialized with Gemini 2.5 Flash")
    
    def process(self, user_input: str, force_merge: bool = False) -> Dict[str, Any]:
        """
        Process a user input through the agent workflow.
        
        Args:
            user_input: The user's input message
            force_merge: If True, merge conflicts without blocking (user accepted changes)
            
        Returns:
            Dictionary containing response and metadata
        """
        initial_state: GraphState = {
            "user_input": user_input,
            "memory_manager": self.memory_manager,
            "global_state": None,
            "extraction_result": None,
            "detected_conflicts": [],
            "augmented_prompt": None,
            "llm_response": None,
            "final_response": None,
            "needs_clarification": False,
            "force_merge": force_merge,
            "merged_fields": [],
            "error": None,
            "langsmith_run_id": None
        }
        
        result = self.graph.invoke(initial_state)
        
        # Get langsmith URL for viewing
        langsmith_url = None
        run_id = result.get("langsmith_run_id")
        if run_id:
            langsmith_url = langsmith.get_run_url(run_id)
        
        return {
            "response": result.get("final_response", ""),
            "extraction": result.get("extraction_result"),
            "conflicts": result.get("detected_conflicts", []),
            "merged_fields": result.get("merged_fields", []),
            "force_merge": force_merge,
            "state": self.memory_manager.state_map,
            "error": result.get("error"),
            "langsmith_run_id": run_id,
            "langsmith_url": langsmith_url
        }
    
    def get_state(self) -> GlobalStateMap:
        """Get the current global state map."""
        return self.memory_manager.state_map
    
    def reset_state(self) -> GlobalStateMap:
        """Reset the state map to defaults."""
        return self.memory_manager.reset_session()
    
    def get_state_summary(self) -> str:
        """Get a formatted summary of the current state."""
        return format_state_summary(self.memory_manager.state_map)
    
    def get_langsmith_url(self) -> Optional[str]:
        """Get the langsmith project URL."""
        if langsmith.client:
            return f"https://smith.langchain.com/o/default/projects/p/{langsmith.PROJECT}"
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_agent(
    state_dir: str = "./data/state",
    chroma_dir: str = "./data/chroma_db"
) -> SSMAgent:
    """
    Create and return a new SSM Agent instance.
    
    Args:
        state_dir: Directory for state persistence
        chroma_dir: Directory for ChromaDB storage
        
    Returns:
        A configured SSMAgent instance
    """
    return SSMAgent(state_dir=state_dir, chroma_dir=chroma_dir)


# =============================================================================
# INITIALIZE ON MODULE LOAD
# =============================================================================

# Initialize langsmith when module is imported
#langsmith.initialize()


# Export all
__all__ = [
    'LLMConfig',
    'langsmithConfig',
    'GeminiClient',
    'SSMAgent',
    'build_agent_graph',
    'create_agent',
]
