"""
SSM Framework - Source Package
"""

from .schema import (
    LanguageEnum,
    ParadigmEnum,
    ToneEnum,
    LogLevel,
    TechState,
    StyleState,
    ProjectContext,
    ConversationMemory,
    ConflictRecord,
    ExtractionResult,
    GlobalStateMap,
    AgentState,
)

from .agents import (
    LLMConfig,
    langsmithConfig,
    GeminiClient,
    SSMAgent,
    create_agent,
)

# Provide a more conventional export name for external consumers
# while preserving the internal lowercase class name.
LangSmithConfig = langsmithConfig

from .memory import (
    ConversationTurn,
    StateMapManager,
    EpisodicMemoryStore,
    MemoryManager,
)

from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    CONFLICT_DETECTION_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT_TEMPLATE,
    build_extraction_prompt,
    build_augmented_prompt,
    build_clarification_prompt,
    format_state_summary,
)

__all__ = [
    # Schema exports
    'LanguageEnum',
    'ParadigmEnum',
    'ToneEnum',
    'LogLevel',
    'TechState',
    'StyleState',
    'ProjectContext',
    'ConversationMemory',
    'ConflictRecord',
    'ExtractionResult',
    'GlobalStateMap',
    'AgentState',
    
    # Agent exports
    'LLMConfig',
    'langsmithConfig',
    'LangSmithConfig',
    'GeminiClient',
    'SSMAgent',
    'create_agent',
    
    # Memory exports
    'ConversationTurn',
    'StateMapManager',
    'EpisodicMemoryStore',
    'MemoryManager',
    
    # Prompt exports
    'EXTRACTION_SYSTEM_PROMPT',
    'CONFLICT_DETECTION_SYSTEM_PROMPT',
    'GENERATION_SYSTEM_PROMPT_TEMPLATE',
    'build_extraction_prompt',
    'build_augmented_prompt',
    'build_clarification_prompt',
    'format_state_summary',
]
