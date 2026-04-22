"""
SSM Framework - State Schema Definitions
========================================

This module defines the comprehensive Pydantic models that form the backbone
of the Sequential-State Management framework. These schemas ensure that all
LLM outputs conform to a strict JSON structure, enabling deterministic state
management and conflict detection.

The GlobalStateMap serves as our  "Source of Truth" for the entire system,
capturing technical constraints, stylistic preferences, and project metadata
that must be preserved across conversation turns.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal, ClassVar
from enum import Enum
from datetime import datetime
import json


class LanguageEnum(str, Enum):
    """Supported programming languages for the project - language agnostic."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    PHP = "php"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    RUBY = "ruby"
    OTHER = "other"


class ParadigmEnum(str, Enum):
    """Programming paradigms that can be enforced."""
    FUNCTIONAL = "functional"
    OBJECT_ORIENTED = "object_oriented"
    PROCEDURAL = "procedural"
    DECLARATIVE = "declarative"
    MIXED = "mixed"


class ToneEnum(str, Enum):
    """Communication tone options."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"


class LogLevel(str, Enum):
    """Logging levels for the system."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TechState(BaseModel):
    """
    Technical constraints and preferences for the project.
    
    This model captures all technical aspects of the project that must be
    preserved throughout the conversation, including language choice,
    frameworks, architectural patterns, and coding standards.
    """
    language: Optional[LanguageEnum] = Field(
        default=None,
        description="Primary programming language (supports all languages) - None until user specifies"
    )
    version: Optional[str] = Field(
        default=None,
        description="Language version (e.g., '3.11' for Python)"
    )
    frameworks: List[str] = Field(
        default_factory=list,
        description="List of frameworks being used (e.g., ['FastAPI', 'React'])"
    )
    libraries: List[str] = Field(
        default_factory=list,
        description="Key libraries and dependencies"
    )
    databases: List[str] = Field(
        default_factory=list,
        description="Databases and data storage systems (e.g., ['PostgreSQL', 'MongoDB', 'Redis', 'DynamoDB'])"
    )
    paradigm: Optional[ParadigmEnum] = Field(
        default=None,
        description="Primary programming paradigm to follow - None until user specifies"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Hard technical constraints that must not be violated"
    )
    architectural_pattern: Optional[str] = Field(
        default=None,
        description="Architectural pattern (e.g., 'MVC', 'Microservices', 'Monolith')"
    )
    testing_framework: Optional[str] = Field(
        default=None,
        description="Testing framework to use (e.g., 'pytest', 'jest')"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="Core domain/task features (e.g., ['CNN', 'object_detection', 'REST API', 'sum'])"
    )
    input_output_types: List[str] = Field(
        default_factory=list,
        description="Input/output data types (e.g., ['image', 'JSON', 'binary data', 'real-time stream'])"
    )
    performance_needs: List[str] = Field(
        default_factory=list,
        description="Performance requirements (e.g., ['real-time', 'high accuracy', 'low latency', '<100ms response'])"
    )
    testing_requirements: List[str] = Field(
        default_factory=list,
        description="Testing and quality requirements (e.g., ['pytest', '80% coverage', 'unit tests', 'integration tests'])"
    )
    deployment_runtime: List[str] = Field(
        default_factory=list,
        description="Deployment and runtime info (e.g., ['Docker', 'Kubernetes', 'AWS Lambda', 'GPU required'])"
    )
    ml_specifics: Dict[str, Any] = Field(
        default_factory=dict,
        description="ML-specific details if applicable (e.g., {'model_types': ['CNN', 'LSTM'], 'gpu_required': True, 'framework': 'TensorFlow'})"
    )
    
    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v: List[str]) -> List[str]:
        """Ensure constraints are non-empty strings."""
        return [c.strip() for c in v if c.strip()]
    
    def set_field(self, field_name: str, value: Any) -> None:
        """
        Set a field value with proper type conversion.
        
        Handles:
        - Enum conversion (string -> LanguageEnum/ParadigmEnum)
        - Direct assignment for other types
        """
        if field_name == "language":
            # Convert string to LanguageEnum
            if isinstance(value, str):
                try:
                    value = LanguageEnum(value)
                except (ValueError, KeyError):
                    print(f"[WARN] Invalid language value: {value}")
                    return
            setattr(self, field_name, value)
        elif field_name == "paradigm":
            # Convert string to ParadigmEnum
            if isinstance(value, str):
                try:
                    value = ParadigmEnum(value)
                except (ValueError, KeyError):
                    print(f"[WARN] Invalid paradigm value: {value}")
                    return
            setattr(self, field_name, value)
        else:
            # Direct assignment for other fields
            setattr(self, field_name, value)


class StyleState(BaseModel):
    """
    Code style and formatting preferences.
    
    Captures all stylistic elements of code generation including naming
    conventions, documentation standards, and formatting preferences.
    Only stores values explicitly set by user - no defaults.
    """
    naming_convention: Optional[str] = Field(
        default=None,
        description="Variable and function naming convention (e.g., snake_case)"
    )
    class_naming: Optional[str] = Field(
        default=None,
        description="Class naming convention (e.g., PascalCase)"
    )
    constant_naming: Optional[str] = Field(
        default=None,
        description="Constant naming convention (e.g., UPPER_SNAKE_CASE)"
    )
    docstring_style: Optional[Literal["google", "numpy", "sphinx", "none"]] = Field(
        default=None,
        description="Docstring format style"
    )
    max_line_length: Optional[int] = Field(
        default=None,
        description="Maximum line length for code"
    )
    indent_size: Optional[int] = Field(
        default=None,
        description="Number of spaces for indentation"
    )
    use_type_hints: Optional[bool] = Field(
        default=None,
        description="Whether to use type hints/annotations"
    )
    comments_level: Optional[Literal["minimal", "moderate", "verbose"]] = Field(
        default=None,
        description="Level of code comments to include"
    )
    prefer_async: Optional[bool] = Field(
        default=None,
        description="Prefer asynchronous code patterns where applicable"
    )


class ProjectContext(BaseModel):
    """
    Project-level context and metadata.
    
    Contains high-level information about the project that provides
    essential context for the LLM's decision-making process.
    """
    project_name: str = Field(
        default="Untitled Project",
        description="Name of the project"
    )
    project_type: Optional[str] = Field(
        default=None,
        description="Type of project (e.g., 'Web Application', 'API', 'CLI Tool')"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief project description"
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="Intended users or audience for the project"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="Key features or requirements of the project"
    )
    stakeholders: List[str] = Field(
        default_factory=list,
        description="List of stakeholders or team members"
    )


class ConversationMemory(BaseModel):
    """
    Conversation history and episodic memory structure.
    
    Tracks the history of interactions and maintains episodic memory
    for context-aware responses.
    """
    turn_count: int = Field(
        default=0,
        description="Total number of conversation turns"
    )
    last_extraction: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last state extraction"
    )
    recent_topics: List[str] = Field(
        default_factory=list,
        description="Recently discussed topics for context"
    )
    pending_clarifications: List[str] = Field(
        default_factory=list,
        description="Items requiring user clarification"
    )
    resolved_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Record of resolved state conflicts"
    )


class ConflictRecord(BaseModel):
    """
    Record of a detected conflict and its resolution status.
    
    Used by the conflict detection system to track contradictions
    between current input and established state rules.
    """
    conflict_id: str = Field(
        description="Unique identifier for this conflict"
    )
    field_path: str = Field(
        description="Dot-notation path to the conflicting field (e.g., 'tech.language')"
    )
    existing_value: Any = Field(
        description="Current value in the state map"
    )
    proposed_value: Any = Field(
        description="New value proposed by user input"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level of the conflict"
    )
    status: Literal["detected", "notified", "resolved", "ignored"] = Field(
        default="detected",
        description="Current resolution status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the conflict was detected"
    )
    resolution_note: Optional[str] = Field(
        default=None,
        description="Note on how the conflict was resolved"
    )


class FeatureExtractionWithConfidence(BaseModel):
    """
    Extracted features with confidence scores and category importance weights.
    
    Implements the structured extraction approach with:
    - Confidence scores (explicit/strong/weak mention)
    - Category weights (importance of each feature type)
    - Per-field confidence tracking
    
    This enables agents to make principled decisions about state updates
    based on both extraction confidence and feature importance.
    """
    
    # Category-level importance weights (based on research)
    CATEGORY_WEIGHTS: ClassVar[Dict[str, float]] = {
        "language": 1.0,           # Critical - don't change lightly
        "framework": 0.9,          # Very important - framework lock-in
        "core_task": 0.95,         # Critical - what user is building
        "constraints": 1.0,        # Critical - hard requirements
        "testing": 0.8,            # Important but flexible
        "deployment": 0.8,         # Important but flexible
        "style": 0.6,              # Less important - can adapt
        "ml_specifics": 0.9        # Very important for ML projects
    }
    
    # Confidence levels
    CONFIDENCE_LEVELS: ClassVar[Dict[str, float]] = {
        "explicit": 0.95,          # User explicitly stated it
        "strongly_implied": 0.75,  # Context makes it very likely
        "weakly_implied": 0.45,    # Weak inference, needs confirmation
    }
    
    # Extracted fields with per-field confidence
    language: Optional[tuple[str, float]] = Field(
        default=None,
        description="(value, confidence) - extracted language"
    )
    paradigm: Optional[tuple[str, float]] = Field(
        default=None,
        description="(value, confidence) - extracted programming paradigm"
    )
    version: Optional[tuple[str, float]] = Field(
        default=None,
        description="(value, confidence) - language version"
    )
    frameworks: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(framework, confidence), ...] - extracted frameworks"
    )
    libraries: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(library, confidence), ...] - extracted libraries"
    )
    key_features: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(feature, confidence), ...] - core domain/task features"
    )
    input_output_types: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(type, confidence), ...] - I/O data types"
    )
    performance_needs: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(requirement, confidence), ...] - performance constraints"
    )
    testing_requirements: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(requirement, confidence), ...] - testing needs"
    )
    deployment_runtime: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(requirement, confidence), ...] - deployment/runtime"
    )
    constraints: List[tuple[str, float]] = Field(
        default_factory=list,
        description="[(constraint, confidence), ...] - hard constraints"
    )
    ml_specifics: Optional[tuple[Dict[str, Any], float]] = Field(
        default=None,
        description="({details}, confidence) - ML-specific information"
    )
    
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence across all extracted fields"
    )
    
    def get_high_confidence_items(self, threshold: float = 0.75) -> Dict[str, Any]:
        """Extract only items above confidence threshold."""
        result = {}
        
        if self.language and self.language[1] >= threshold:
            result["language"] = self.language[0]
        if self.version and self.version[1] >= threshold:
            result["version"] = self.version[0]
        
        result["frameworks"] = [f[0] for f in self.frameworks if f[1] >= threshold]
        result["libraries"] = [l[0] for l in self.libraries if l[1] >= threshold]
        result["key_features"] = [k[0] for k in self.key_features if k[1] >= threshold]
        result["input_output_types"] = [i[0] for i in self.input_output_types if i[1] >= threshold]
        result["performance_needs"] = [p[0] for p in self.performance_needs if p[1] >= threshold]
        result["testing_requirements"] = [t[0] for t in self.testing_requirements if t[1] >= threshold]
        result["deployment_runtime"] = [d[0] for d in self.deployment_runtime if d[1] >= threshold]
        result["constraints"] = [c[0] for c in self.constraints if c[1] >= threshold]
        
        if self.ml_specifics and self.ml_specifics[1] >= threshold:
            result["ml_specifics"] = self.ml_specifics[0]
        
        return result


class ExtractionResult(BaseModel):
    """
    Result of the state extraction process.
    
    Contains the extracted attributes, confidence scores, chain-of-thought
    reasoning, and any detected conflicts from the extraction phase.
    """
    extracted_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted state attributes from user input"
    )
    chain_of_thought: Optional[str] = Field(
        default=None,
        description="Reasoning process - what the user is building and how features were extracted"
    )
    feature_confidence: Optional[FeatureExtractionWithConfidence] = Field(
        default=None,
        description="Detailed confidence scores and importance weights for extracted features"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the extraction (0-1)"
    )
    raw_intent: str = Field(
        default="",
        description="Raw interpreted intent from user input"
    )
    detected_conflicts: List[ConflictRecord] = Field(
        default_factory=list,
        description="List of conflicts detected during extraction"
    )
    requires_clarification: bool = Field(
        default=False,
        description="Whether user clarification is needed"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Question to ask user if clarification needed"
    )



class GlobalStateMap(BaseModel):
    """
    The complete state map serving as the "Source of Truth".
    
    This is the primary data structure that the SSM framework manages.
    It encapsulates all project-relevant state that must be preserved
    across conversation turns, including technical constraints, style
    preferences, project context, and conversation memory.
    
    The state map is:
    1. Persisted to disk as JSON after each validated update
    2. Used to construct State Anchors for prompt augmentation
    3. Compared against extractions for conflict detection
    """
    tech: TechState = Field(
        default_factory=TechState,
        description="Technical constraints and preferences"
    )
    style: StyleState = Field(
        default_factory=StyleState,
        description="Code style and formatting preferences"
    )
    project: ProjectContext = Field(
        default_factory=ProjectContext,
        description="Project-level context and metadata"
    )
    tone: Optional[ToneEnum] = Field(
        default=None,
        description="Communication tone preference (e.g., professional, casual)"
    )
    memory: ConversationMemory = Field(
        default_factory=ConversationMemory,
        description="Conversation history and episodic memory"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the state map was created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When the state map was last updated"
    )
    version: int = Field(
        default=1,
        description="State map version for tracking updates"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize computed fields after model creation."""
        pass
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization - only persist technical features.
        
        When saving to JSON, only include:
        - tech (all comprehensive technical fields)
        - memory (turn tracking, conflict history)
        
        Excludes: style preferences, tone, project context
        """
        data = super().model_dump(**kwargs)
        
        # Persist comprehensive technical stack - EXCLUDE None values
        tech_fields = data.get("tech", {})
        cleaned = {
            "tech": {
                # Only include language if explicitly set (not None)
                **({"language": tech_fields.get("language")} if tech_fields.get("language") else {}),
                # Only include version if present
                **({"version": tech_fields.get("version")} if tech_fields.get("version") else {}),
                "frameworks": tech_fields.get("frameworks", []),
                "libraries": tech_fields.get("libraries", []),
                "databases": tech_fields.get("databases", []),
                "key_features": tech_fields.get("key_features", []),
                "input_output_types": tech_fields.get("input_output_types", []),
                "performance_needs": tech_fields.get("performance_needs", []),
                "testing_requirements": tech_fields.get("testing_requirements", []),
                "deployment_runtime": tech_fields.get("deployment_runtime", []),
                "ml_specifics": tech_fields.get("ml_specifics", {}),
                # Only include paradigm if explicitly set (not None)
                **({"paradigm": tech_fields.get("paradigm")} if tech_fields.get("paradigm") else {}),
                # Only include architectural_pattern if present
                **({"architectural_pattern": tech_fields.get("architectural_pattern")} if tech_fields.get("architectural_pattern") else {}),
                # Only include testing_framework if present
                **({"testing_framework": tech_fields.get("testing_framework")} if tech_fields.get("testing_framework") else {}),
                "constraints": tech_fields.get("constraints", [])
            },
            "memory": {
                "turn_count": data.get("memory", {}).get("turn_count", 0),
                "resolved_conflicts": data.get("memory", {}).get("resolved_conflicts", [])
            }
        }
        
        return cleaned
    
    def to_anchor_string(self) -> str:
        """
        Convert the state map to a formatted string for prompt anchoring.
        
        This method generates a human-readable representation of the current
        state that can be injected into LLM prompts as a "State Anchor",
        ensuring the LLM has full context of established constraints.
        
        IMPORTANT: Only includes EXPLICITLY SET constraints, skipping None/empty values.
        This prevents uninitialized defaults from being treated as constraints.
        """
        sections = []
        
        # Technical State - ONLY include if explicitly set
        tech_items = []
        
        # Language - only if explicitly set (not None)
        if self.tech.language:
            tech_items.append(f"Language: {self.tech.language.value}")
        
        if self.tech.version:
            tech_items.append(f"Version: {self.tech.version}")
        
        if self.tech.frameworks:
            tech_items.append(f"Frameworks: {', '.join(self.tech.frameworks)}")
        
        if self.tech.libraries:
            tech_items.append(f"Libraries: {', '.join(self.tech.libraries[:10])}")
        
        # Paradigm - only if explicitly set (not None)
        if self.tech.paradigm:
            tech_items.append(f"Paradigm: {self.tech.paradigm.value}")
        
        if self.tech.architectural_pattern:
            tech_items.append(f"Architecture: {self.tech.architectural_pattern}")
        
        if self.tech.testing_framework:
            tech_items.append(f"Testing: {self.tech.testing_framework}")
        
        if self.tech.constraints:
            tech_items.append("Hard Constraints:")
            for c in self.tech.constraints:
                tech_items.append(f"  - {c}")
        
        # Only add technical section if there are items
        if tech_items:
            sections.append("=== TECHNICAL CONSTRAINTS ===")
            sections.extend(tech_items)
        else:
            sections.append("=== TECHNICAL CONSTRAINTS ===")
            sections.append("(None established yet - will be set from your first request)")
        
        # Style State - ONLY include if explicitly set
        style_items = []
        if self.style.naming_convention:
            style_items.append(f"Naming: {self.style.naming_convention} (variables/functions)")
        if self.style.class_naming:
            style_items.append(f"Class Naming: {self.style.class_naming}")
        if self.style.docstring_style:
            style_items.append(f"Docstrings: {self.style.docstring_style}")
        if self.style.max_line_length:
            style_items.append(f"Line Length: {self.style.max_line_length}")
        if self.style.use_type_hints is not None:
            style_items.append(f"Type Hints: {'Yes' if self.style.use_type_hints else 'No'}")
        if self.style.comments_level:
            style_items.append(f"Comment Level: {self.style.comments_level}")
        if self.style.indent_size:
            style_items.append(f"Indent Size: {self.style.indent_size}")
        if self.style.prefer_async is not None:
            style_items.append(f"Prefer Async: {'Yes' if self.style.prefer_async else 'No'}")
        
        if style_items:
            sections.append("\n=== CODE STYLE ===")
            sections.extend(style_items)
        else:
            sections.append("\n=== CODE STYLE ===")
            sections.append("(None established yet - will be set from your requests)")
        
        # Project Context
        sections.append("\n=== PROJECT CONTEXT ===")
        sections.append(f"Name: {self.project.project_name}")
        if self.project.project_type:
            sections.append(f"Type: {self.project.project_type}")
        if self.project.description:
            sections.append(f"Description: {self.project.description}")
        if self.project.key_features:
            sections.append(f"Key Features: {', '.join(self.project.key_features[:5])}")
        
        # Communication - ONLY include tone if explicitly set
        if self.tone:
            sections.append(f"\n=== COMMUNICATION ===")
            sections.append(f"Tone: {self.tone.value if hasattr(self.tone, 'value') else self.tone}")
        else:
            sections.append(f"\n=== COMMUNICATION ===")
            sections.append("(Tone not yet established - will adapt to your style)")
        
        return "\n".join(sections)
    
    def to_json_file(self, filepath: str) -> None:
        """Save the state map to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Use model_dump (which excludes None values) then convert to JSON
            cleaned_data = self.model_dump(exclude_none=False)
            json.dump(cleaned_data, f, indent=2, default=str)  # default=str for datetime serialization
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'GlobalStateMap':
        """Load a state map from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # model_validate will apply defaults for missing fields
        return cls.model_validate(data)
    
    def increment_version(self) -> None:
        """Increment the version counter and update timestamp."""
        self.version += 1
        self.last_updated = datetime.now()
    
    def merge_extraction(
        self, 
        extraction: ExtractionResult,
        force: bool = False,
        confidence_threshold: float = 0.65
    ) -> List[ConflictRecord]:
        """
        Merge technical features from extraction into state map using confidence scores.
        
        Uses category weights and confidence scores to make principled decisions
        about which extractions to merge and when to flag conflicts.
        
        Args:
            extraction: The extraction result to merge
            force: If True, override conflicts without notification
            confidence_threshold: Minimum confidence (0-1) to merge field updates
            
        Returns:
            List of conflicts detected during the merge
        """
        conflicts = []
        extracted = extraction.extracted_state
        
        # Category weights for severity assignment
        category_weights = {
            'language': 1.0,
            'paradigm': 1.0,
            'version': 0.9,
            'frameworks': 0.9,
            'libraries': 0.8,
            'databases': 0.9,
            'key_features': 0.95,
            'input_output_types': 0.85,
            'performance_needs': 1.0,
            'testing_requirements': 0.8,
            'deployment_runtime': 0.8,
            'ml_specifics': 0.9,
            'constraints': 1.0
        }
        
        # ONLY handle tech state updates - ignore style, tone, project
        if 'tech' in extracted:
            tech_updates = extracted['tech']
            for field, new_value in tech_updates.items():
                # Only allow known technical fields
                allowed_fields = list(category_weights.keys())
                if field not in allowed_fields:
                    continue
                
                # Check confidence if feature_confidence is available
                field_confidence = extraction.confidence
                if extraction.feature_confidence:
                    # Get per-field confidence (varies by field type)
                    if field == 'language' and extraction.feature_confidence.language:
                        field_confidence = extraction.feature_confidence.language[1]
                    elif field == 'paradigm' and extraction.feature_confidence.paradigm:
                        field_confidence = extraction.feature_confidence.paradigm[1]
                    elif field == 'version' and extraction.feature_confidence.version:
                        field_confidence = extraction.feature_confidence.version[1]
                    elif field == 'frameworks' and extraction.feature_confidence.frameworks:
                        field_confidence = sum(f[1] for f in extraction.feature_confidence.frameworks) / len(extraction.feature_confidence.frameworks) if extraction.feature_confidence.frameworks else 0
                    elif field == 'constraints' and extraction.feature_confidence.constraints:
                        field_confidence = sum(c[1] for c in extraction.feature_confidence.constraints) / len(extraction.feature_confidence.constraints) if extraction.feature_confidence.constraints else 0
                    # ... add other fields as needed
                
                # Skip if below confidence threshold
                if field_confidence < confidence_threshold:
                    continue
                    
                if hasattr(self.tech, field):
                    current_value = getattr(self.tech, field)
                    
                    # Check for conflicts
                    if current_value != new_value and current_value is not None and current_value != [] and current_value != "":
                        # Determine severity based on category weight
                        weight = category_weights.get(field, 0.5)
                        severity = "critical" if weight >= 0.95 else ("high" if weight >= 0.8 else "medium")
                        
                        conflict = ConflictRecord(
                            conflict_id=f"tech.{field}_{datetime.now().timestamp()}",
                            field_path=f"tech.{field}",
                            existing_value=str(current_value),
                            proposed_value=str(new_value),
                            severity=severity,
                            status="ignored" if force else "detected"
                        )
                        conflicts.append(conflict)
                        if force:
                            # Properly set the field based on its type
                            self.tech.set_field(field, new_value)
                    elif current_value is None or current_value == [] or current_value == "":
                        # Set if currently empty - validate through field setter
                        self.tech.set_field(field, new_value)
        
        # Update memory
        self.memory.turn_count += 1
        self.memory.last_extraction = datetime.now()
        
        if conflicts:
            self.memory.resolved_conflicts.extend([c.model_dump() for c in conflicts])
        
        if not conflicts or force:
            self.increment_version()
        
        return conflicts


class AgentState(BaseModel):
    """
    State container for the LangGraph agent workflow.
    
    This model represents the complete state passed between nodes
    in the LangGraph execution graph, including the global state map,
    current input, extraction results, and generated responses.
    """
    global_state: GlobalStateMap = Field(
        default_factory=GlobalStateMap,
        description="The persistent global state map"
    )
    user_input: str = Field(
        default="",
        description="Current user input"
    )
    extraction_result: Optional[ExtractionResult] = Field(
        default=None,
        description="Result of state extraction"
    )
    detected_conflicts: List[ConflictRecord] = Field(
        default_factory=list,
        description="Conflicts detected in this turn"
    )
    augmented_prompt: Optional[str] = Field(
        default=None,
        description="Final prompt with state anchors"
    )
    llm_response: Optional[str] = Field(
        default=None,
        description="Raw response from the LLM"
    )
    final_response: Optional[str] = Field(
        default=None,
        description="Final response to send to user"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether user clarification is needed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )
    langsmith_run_id: Optional[str] = Field(
        default=None,
        description="LangSmith run ID for tracing"
    )


# Export all models
__all__ = [
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
]
