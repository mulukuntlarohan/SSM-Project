"""
SSM Framework - Prompt Templates Module
=======================================

This module contains all system instructions, prompt templates, and
prompt construction utilities for the SSM framework. It handles most of the promts:

1. Extraction prompts for parsing user intent into structured state
2. Conflict detection prompts for identifying contradictions
3. State anchor construction for prompt augmentation
4. Final generation prompts with embedded constraints

The prompt engineering follows a consistent pattern:
- Clear role definition for the LLM
- Structured output requirements (JSON schema)
- Few-shot examples where beneficial
- Explicit constraint enforcement instructions
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import GlobalStateMap, ExtractionResult, ConflictRecord


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are the Technical Feature Extraction Engine for the SSM framework.

Your role is to analyze user input and extract comprehensive technical features with confidence scores.
Use CHAIN-OF-THOUGHT reasoning to think through what the user is asking for, then extract features.

## CHAIN-OF-THOUGHT REASONING PROCESS (think through this)

1. **Understand the Context**: What is the user trying to build? (web app, API, CLI, game, data tool, etc.)
2. **Identify the Language**: What programming language are they using? (Python, JavaScript, Java, C#, Go, Rust, C++, Ruby, PHP, etc.)
3. **Extract the Stack**: What frameworks, libraries, technologies are mentioned?
4. **Determine the Domain**: Is this frontend, backend, full-stack, mobile, data/ML, systems, game dev, etc.?
5. **Extract Requirements**: What are the constraints, performance needs, deployment targets?
6. **Assess Confidence**: How certain am I about each extracted item?

## Supported Languages (Any Programming Language)

Frontend: JavaScript, TypeScript, HTML, CSS, React, Vue, Angular, Svelte, Astro
Backend: Python, Node.js, Java, C#, Go, Rust, Ruby, PHP, Kotlin, Scala
Databases: SQL, PostgreSQL, MySQL, MongoDB, Redis, Firebase, DynamoDB, Cassandra
Systems: C, C++, Rust, Go, Assembly
Mobile: Swift, Kotlin, React Native, Flutter, Dart
Other: Bash, PowerShell, Groovy, Elixir, Haskell, Clojure, Julia, R

## Category Importance Weights

These weights guide severity assignment for conflicts:
- Language: 1.0 (critical - don't change)
- Core Task/Domain: 0.95 (critical - what they're building)
- Constraints: 1.0 (critical - hard requirements)
- Framework: 0.9 (very important)
- Database/Backend: 0.9 (very important)
- Deployment: 0.8 (important but flexible)
- Testing: 0.8 (important but flexible)
- Style: 0.6 (less important - easily adapted)

## Confidence Levels

Score your extraction confidence:
- **Explicit (0.95)**: User stated it directly ("build in React with TypeScript")
- **Strongly Implied (0.75)**: Context makes it very likely ("REST API" implies backend/server)
- **Weakly Implied (0.45)**: Weak inference, needs confirmation ("web project" implies frontend/backend but unclear)

## What to Extract (Universal across all domains)

Always check for these fields (extract only if mentioned or strongly implied):

1. **Programming Language(s)** - Normalize to: python, javascript, typescript, java, csharp, go, rust, cpp, php, kotlin, swift, ruby, sql, other
   - NOTE: "C++" → "cpp", "C#" → "csharp", "Node.js" → "javascript"
2. **Language Version** - specific version if mentioned (3.11, 18.0, .NET 8, etc.)
3. **Frameworks & Libraries** - React, Vue, FastAPI, Spring, Django, Rails, Express, .NET Core, Svelte, Vue, Angular, etc.
4. **Databases/Data Storage** - PostgreSQL, MongoDB, Redis, Firebase, MySQL, DynamoDB, etc.
5. **Core Domain/Features** - what they're building: web app, REST API, CLI tool, game, data pipeline, mobile app, dashboard, real-time chat, ecommerce, CMS, etc.
6. **Frontend/Backend/Full-Stack** - type of application (frontend only, backend only, full-stack, mobile, desktop, etc.)
7. **Input/Output Types** - data types: JSON, CSV, images, video, streams, files, databases, APIs, websockets, etc.
8. **Performance Needs** - requirements: real-time, low latency, <100ms response, high throughput, etc.
9. **Testing Requirements** - constraints: Jest, pytest, unittest, RSpec, JUnit, >80% coverage, E2E tests, etc.
10. **Deployment/Runtime** - context: Docker, Kubernetes, AWS, Azure, Vercel, Heroku, on-premise, edge, mobile, etc.
11. **Hard Constraints** - non-negotiable requirements: must use X, cannot use Y, thread-safe, ACID compliant, etc.

## What to IGNORE (Unless Explicitly Requested)
- Tone/politeness (formal, casual, friendly) - ONLY extract if user says "professional tone" or similar
- Naming conventions (snake_case, camelCase) - ONLY extract if user says "use snake_case variables"
- Code formatting (line length, indentation) - ONLY extract if user mentions specific formatting
- Documentation preferences (docstrings, comments) - ONLY extract if user specifies docs style
- Personal style preferences - ONLY extract if explicitly mentioned
- Non-technical chitchat

## OUTPUT SCHEMA (MUST FOLLOW EXACTLY)

Always return JSON matching this strict schema:
```json
{
  "chain_of_thought": "Your reasoning about what user is building and technologies",
  "extracted_state": {
    "tech": {
      "language": "REQUIRED: One of [python, javascript, typescript, java, csharp, go, rust, cpp, php, kotlin, swift, ruby, sql, other]. MUST be lowercase.",
      "version": "optional: version string like '3.11' or '17'",
      "frameworks": "optional: list of framework names",
      "databases": "optional: list of database names",
      "libraries": "optional: list of library names",
      "key_features": "optional: list of what they're building (examples: 'snake game', 'REST API', 'web scraper')",
      "paradigm": "optional: functional|imperative|oop|procedural",
      "deployment_runtime": "optional: list of deployment targets",
      "constraints": "optional: list of hard constraints",
      "input_output_types": "optional: list of data types",
      "performance_needs": "optional: latency/throughput requirements",
      "testing_requirements": "optional: testing framework/coverage",
      "architectural_pattern": "optional: mvc|microservices|monolith|serverless"
    }
  },
  "raw_intent": "User's original intent in 1-2 sentences",
  "confidence": "0.0-1.0: confidence score for this extraction",
  "requires_clarification": false
}
```

**CRITICAL LANGUAGE NORMALIZATION**:
- If user says "C++", return "cpp"
- If user says "C#", return "csharp"  
- If user says "Java", return "java"
- If user says "Python", return "python"
- If user says anything unclear, return "other" (NOT language name)
- Always use LOWERCASE
- The language field is MANDATORY - never omit it

**NEVER include tone, style, or project fields** - those are not part of extracted_state

## Output Format

Return VALID JSON with all extracted fields. Omit null values unless they're critical context.

```json
{
  "chain_of_thought": "User wants a full-stack web application: React frontend for dashboard with real-time updates, Node.js/Express backend API connecting to PostgreSQL database. Must handle 1000+ concurrent users and deploy to AWS. Needs JWT auth and responsive design.",
  "extracted_state": {
    "tech": {
      "language": ["javascript", "sql"],
      "version": null,
      "frameworks": ["react", "express", "postgresql"],
      "libraries": ["socket.io", "jwt"],
      "key_features": ["real-time dashboard", "REST API", "authentication", "responsive design"],
      "input_output_types": ["JSON", "WebSocket", "database"],
      "performance_needs": ["<500ms response", "1000+ concurrent", "real-time updates"],
      "testing_requirements": ["jest", "integration tests"],
      "deployment_runtime": ["Docker", "AWS EC2"],
      "constraints": ["scalable", "production-ready"]
    }
  },
  "raw_intent": "Full-stack real-time dashboard using React frontend and Node.js/Express API with PostgreSQL, deployable to AWS",
  "confidence": 0.93
}
```

## Diverse Examples (Not just ML/Python)

### Example 1: Frontend Web Development
**Input**: "Make a responsive React component library with TypeScript and Tailwind CSS for design system."

**CoT**: User wants frontend components, specifically React library with TypeScript for type safety and Tailwind for styling.

**Output**:
```json
{
  "chain_of_thought": "Frontend-focused project. React is the framework, TypeScript for safety, Tailwind for styling. This is a component library, not backend. No database needed.",
  "extracted_state": {
    "tech": {
      "language": "javascript",
      "frameworks": ["react", "typescript", "tailwind"],
      "key_features": ["component library", "design system", "responsive"],
      "testing_requirements": ["storybook", "jest", "visual tests"],
      "deployment_runtime": ["npm", "CDN"]
    }
  },
  "raw_intent": "Frontend React component library with TypeScript and Tailwind for design system",
  "confidence": 0.97
}
```

### Example 2: Backend/Database
**Input**: "Java microservice with Spring Boot connecting to PostgreSQL, runs in Docker on Kubernetes cluster."

**CoT**: Backend-focused. Java with Spring framework, structured database (PostgreSQL), containerized deployment. This is not a web interface project.

**Output**:
```json
{
  "chain_of_thought": "Backend microservice. Java + Spring framework. Database is PostgreSQL (relational). Deployment: Docker → Kubernetes. This is infrastructure/backend focused.",
  "extracted_state": {
    "tech": {
      "language": "java",
      "version": "17+",
      "frameworks": ["spring-boot", "spring-jpa"],
      "databases": ["postgresql"],
      "key_features": ["microservice", "REST API", "persistence"],
      "deployment_runtime": ["docker", "kubernetes"]
    }
  },
  "raw_intent": "Java Spring Boot microservice with PostgreSQL, containerized for Kubernetes",
  "confidence": 0.96
}
```

### Example 3: Full-Stack Web App
**Input**: "Build an ecommerce site with Next.js frontend, Node.js backend API, MongoDB for products, Redis for caching, and Stripe for payments."

**CoT**: Full-stack project. Next.js for frontend+backend (modern framework). MongoDB for flexible schema. Redis for performance. Stripe for payment processing. Need auth, product catalog, cart, checkout.

**Output**:
```json
{
  "chain_of_thought": "Full-stack ecommerce. Next.js provides both frontend and backend. MongoDB for product data (NoSQL). Redis for caching layer. Stripe integration for payments. Need user auth, product search, cart management.",
  "extracted_state": {
    "tech": {
      "language": "javascript",
      "frameworks": ["next.js", "react"],
      "databases": ["mongodb", "redis"],
      "libraries": ["stripe-api"],
      "key_features": ["ecommerce", "payments", "product catalog", "shopping cart", "user auth"],
      "input_output_types": ["JSON", "PaymentGateway"],
      "deployment_runtime": ["vercel", "docker"]
    }
  },
  "raw_intent": "Full-stack ecommerce platform with Next.js, MongoDB, Redis caching, and Stripe payments",
  "confidence": 0.95
}
```

### Example 4: Systems Programming
**Input**: "Write a low-level networking library in Rust with async support, must be thread-safe and zero-copy where possible."

**CoT**: Systems programming. Rust language. Async is about concurrency/performance. Thread-safe and zero-copy are hard constraints. This is a library, not application.

**Output**:
```json
{
  "chain_of_thought": "Low-level systems library in Rust. Async programming for concurrency. Key constraints: thread-safety (Rust guarantees this) and zero-copy (performance critical). This is a library, not an application.",
  "extracted_state": {
    "tech": {
      "language": "rust",
      "frameworks": ["tokio"],
      "key_features": ["networking", "async", "thread-safe", "low-level"],
      "performance_needs": ["zero-copy", "high throughput", "low latency"],
      "constraints": ["thread-safe", "memory-safe", "performance-critical"]
    }
  },
  "raw_intent": "Thread-safe, zero-copy async networking library in Rust",
  "confidence": 0.96
}
```

### Example 5: Data/Analytics Project
**Input**: "ETL pipeline in Python using Apache Airflow to ingest data from PostgreSQL, transform with Pandas, load to Snowflake data warehouse."

**CoT**: Data engineering. Python + Airflow for orchestration. PostgreSQL source, Pandas for ETL, Snowflake sink. This is infrastructure/pipeline focused, not web or ML model training.

**Output**:
```json
{
  "chain_of_thought": "Data engineering pipeline. Python for orchestration. Airflow schedules/monitors jobs. PostgreSQL is source system. Pandas for data transformation. Snowflake for analytics warehouse. This is ETL/infrastructure, not web or ML model.",
  "extracted_state": {
    "tech": {
      "language": "python",
      "frameworks": ["apache-airflow", "pandas"],
      "databases": ["postgresql", "snowflake"],
      "key_features": ["ETL", "data pipeline", "orchestration"],
      "input_output_types": ["database", "CSV", "data warehouse"],
      "deployment_runtime": ["docker", "on-premise", "cloud"]
    }
  },
  "raw_intent": "ETL pipeline: PostgreSQL → Pandas transform → Snowflake using Apache Airflow",
  "confidence": 0.95
}
```

## Critical Rules

1. **Use chain-of-thought first** - Think through what the user is building before extracting
2. **Extract only what's mentioned or strongly implied** - Don't over-infer
3. **Include all relevant fields** - Even if empty, show the structure
4. **Confidence reflects certainty** - Explicit=0.95+, Strongly-Implied=0.70-0.75, Weak=0.45-0.50
5. **Return valid JSON** - All values properly formatted, no syntax errors
6. **Support ANY programming language** - Not just Python, include all: C, C++, C#, Java, Go, Rust, Ruby, PHP, Swift, Kotlin, etc.
7. **Ignore personal preferences** - Focus on technical stack and requirements, not style
"""

CONFLICT_DETECTION_SYSTEM_PROMPT = """You are the Conflict Detection Engine for the SSM framework. Your role is to identify contradictions between user input and established project constraints.

## Types of Conflicts

1. **Critical Conflicts**: Direct contradictions of hard constraints
   - Language changes (Python → Java)
   - Paradigm changes (Functional → OOP)
   - Framework replacements (React → Vue)

2. **High Severity**: Violations of architectural patterns
   - Breaking established architectural patterns
   - Ignoring defined testing strategies
   - Contradicting security requirements

3. **Medium Severity**: Style inconsistencies
   - Naming convention changes
   - Documentation style changes
   - Code organization preferences

4. **Low Severity**: Minor preference changes
   - Indentation size changes
   - Line length adjustments
   - Comment density changes

## Conflict Resolution Protocol

When a conflict is detected:
1. Notify the user with a clear explanation
2. Ask for confirmation before proceeding
3. Provide options: "Proceed anyway", "Keep existing", "Clarify"
4. Log the conflict for future reference

## Output Format

Return a JSON object with conflict details:
```json
{
  "conflicts_found": true/false,
  "conflict_list": [
    {
      "field": "tech.language",
      "existing": "python",
      "proposed": "java",
      "severity": "critical",
      "recommendation": "Ask user to confirm language change"
    }
  ],
  "resolution_suggestion": "string"
}
```
"""

GENERATION_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant operating under the Sequential-State Management (SSM) framework. You are helping with a software development project and must strictly adhere to established constraints.

## CRITICAL: MANDATORY CONSTRAINTS

⚠️ **Language Requirement**: You MUST generate code ONLY in the language specified in the State Anchor below.
If the language is set to 'java', write Java code. If 'python', write Python code. Never violate the language constraint.

## CRITICAL: State Anchor

The following constraints have been established for this project. You MUST follow these rules in all your responses:

{state_anchor}

## Your Responsibilities

1. **Language Adherence**: Generate code EXCLUSIVELY in the specified language - this is non-negotiable
2. **Constraint Adherence**: Never violate the hard constraints listed above
3. **Style Consistency**: Match the code style and communication tone specified
4. **Context Awareness**: Consider the project context in all responses
5. **Memory Preservation**: Remember established rules across the conversation

## Response Guidelines

- Generate code ONLY in the specified language with no exceptions
- Follow the established naming conventions exactly
- Include documentation in the specified format
- Maintain the specified communication tone
- Reference established constraints when relevant
- If a request conflicts with the language constraint, regenerate in the correct language

## When In Doubt

If a user request would violate established constraints:
1. Explain the conflict clearly
2. Suggest alternatives that comply with constraints
3. Ask for clarification if needed
4. Never silently ignore established rules
"""


# =============================================================================
# PROMPT CONSTRUCTION FUNCTIONS
# =============================================================================

def build_extraction_prompt(
    user_input: str,
    current_state: GlobalStateMap,
    relevant_history: Optional[List[str]] = None
) -> str:
    """
    Build the extraction prompt with chain-of-thought reasoning.
    
    Guides the LLM to think through what the user is asking before
    extracting technical features. Supports all programming languages
    and domains.
    
    Args:
        user_input: The user's input message
        current_state: The current global state map
        relevant_history: Optional list of relevant past conversation turns
        
    Returns:
        The complete extraction prompt with CoT instructions
    """
    prompt_parts = [
        "## CHAIN-OF-THOUGHT ANALYSIS",
        "Before extracting features, think through these questions:",
        "1. What is the user trying to build? (web app, API, CLI, library, data pipeline, etc.)",
        "2. What programming language(s) are they using? (Python, JavaScript, Java, C#, Go, Rust, SQL, Ruby, etc.)",
        "3. What frameworks/libraries are they mentioning?",
        "4. Is this frontend, backend, full-stack, mobile, or system-level code?",
        "5. What are the specific constraints or requirements mentioned?",
        "",
        "## Current State",
        "Here is the current project state that you should compare against:",
        "```json",
        current_state.model_dump_json(indent=2),
        "```",
        ""
    ]
    
    if relevant_history:
        prompt_parts.extend([
            "## Recent Context",
            "Here are relevant past conversation segments (consider context from prior turns):",
        ])
        for i, hist in enumerate(relevant_history[:3]):
            prompt_parts.append(f"{i+1}. {hist}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        "## User Input",
        "Analyze the following user input carefully:",
        "",
        f'"{user_input}"',
        "",
        "STEP 1: Think-aloud - what is the user building and what technologies are involved?",
        "STEP 2: Extract all relevant technical features with confidence scores.",
        "STEP 3: Return valid JSON with chain_of_thought field showing your reasoning.",
        ""
    ])
    
    return "\n".join(prompt_parts)


def build_conflict_detection_prompt(
    extraction_result: ExtractionResult,
    current_state: GlobalStateMap
) -> str:
    """
    Build the conflict detection prompt.
    
    Args:
        extraction_result: The result of state extraction
        current_state: The current global state map
        
    Returns:
        The complete conflict detection prompt
    """
    prompt_parts = [
        "## Task: Conflict Detection",
        "",
        "Compare the extracted state changes against the current state:",
        "",
        "### Current State:",
        "```json",
        current_state.model_dump_json(indent=2),
        "```",
        "",
        "### Proposed Changes:",
        "```json",
        extraction_result.model_dump_json(indent=2),
        "```",
        "",
        "Identify any conflicts between the proposed changes and existing state.",
        "Return a JSON object with conflict details."
    ]
    
    return "\n".join(prompt_parts)


def build_augmented_prompt(
    user_input: str,
    state_map: GlobalStateMap,
    relevant_context: Optional[List[str]] = None
) -> str:
    """
    Build the final augmented prompt with state anchors.
    
    This function constructs the complete prompt that will be sent to the
    target LLM, including the state anchor that ensures constraint adherence.
    
    Args:
        user_input: The user's input message
        state_map: The current global state map
        relevant_context: Optional relevant past conversation segments
        
    Returns:
        The complete augmented prompt
    """
    # Generate state anchor
    state_anchor = state_map.to_anchor_string()
    
    # Build system prompt with state anchor
    system_prompt = GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        state_anchor=state_anchor
    )
    
    # Build context section if available
    context_section = ""
    if relevant_context:
        context_section = "\n## Relevant Context\n" + "\n".join([
            f"- {ctx}" for ctx in relevant_context[:3]
        ])
    
    # Combine into final prompt
    augmented_prompt = f"""{system_prompt}
{context_section}

## Current Request

{user_input}

## Response

Please respond to the user's request while strictly adhering to all established constraints.
"""
    
    return augmented_prompt


def build_clarification_prompt(
    conflicts: List[ConflictRecord],
    user_input: str
) -> str:
    """
    Build a clarification request for conflict resolution.
    
    Args:
        conflicts: List of detected conflicts
        user_input: The original user input
        
    Returns:
        The clarification prompt
    """
    conflict_descriptions = []
    for conflict in conflicts:
        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(conflict.severity, "❓")
        
        conflict_descriptions.append(
            f"{severity_emoji} **{conflict.field_path}**: "
            f"`{conflict.existing_value}` → `{conflict.proposed_value}` "
            f"({conflict.severity} severity)"
        )
    
    prompt = f"""I noticed a potential conflict with your request:

{chr(10).join(conflict_descriptions)}

Your original request was: "{user_input}"

Would you like to:
1. **Proceed** with the new changes (this may affect existing constraints)
2. **Keep** the existing configuration
3. **Clarify** what you actually want

Please let me know how you'd like to proceed.
"""
    
    return prompt


def build_reflexion_prompt(
    llm_response: str,
    state_map: GlobalStateMap,
    original_request: str
) -> str:
    """
    Build a reflexion prompt for response auditing.
    
    This prompt asks the LLM to verify that its own response
    adheres to the established constraints.
    
    Args:
        llm_response: The response to audit
        state_map: The current global state map
        original_request: The original user request
        
    Returns:
        The reflexion prompt
    """
    prompt = f"""## Reflexion Task

Please review your response to verify constraint adherence.

### Original Request:
{original_request}

### Your Response:
{llm_response}

### Constraints to Verify:
{state_map.to_anchor_string()}

### Verification Checklist:
Check if your response:
1. Uses the correct programming language
2. Follows the specified naming conventions
3. Uses the correct documentation style
4. Maintains the appropriate tone
5. Respects all hard constraints

If any violations are found, respond with:
```json
{{
  "violations_found": true,
  "violations": [
    {{
      "constraint": "name of violated constraint",
      "description": "how it was violated",
      "corrected_response": "the corrected version"
    }}
  ],
  "should_regenerate": true
}}
```

If no violations are found:
```json
{{
  "violations_found": false,
  "violations": [],
  "should_regenerate": false
}}
```
"""
    
    return prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_state_summary(state_map: GlobalStateMap) -> str:
    """
    Create a human-readable summary of the current state.
    
    Args:
        state_map: The global state map to summarize
        
    Returns:
        A formatted string summary
    """
    lines = [
        f"📊 **State Map v{state_map.version}**",
        f"Last updated: {state_map.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"🖥️ **Language**: {state_map.tech.language.value if state_map.tech.language else '(Not set)'}",
        f"📐 **Paradigm**: {state_map.tech.paradigm.value if state_map.tech.paradigm else '(Not set)'}",
    ]
    
    if state_map.tech.frameworks:
        lines.append(f"📚 **Frameworks**: {', '.join(state_map.tech.frameworks)}")
    
    if state_map.tech.constraints:
        lines.append(f"🔒 **Constraints**: {len(state_map.tech.constraints)} active")
    
    # Only show style if any style is set
    style_parts = []
    if state_map.style.naming_convention:
        style_parts.append(state_map.style.naming_convention)
    if state_map.style.docstring_style:
        style_parts.append(f"{state_map.style.docstring_style} docs")
    
    if style_parts:
        lines.extend([
            "",
            f"✨ **Style**: {', '.join(style_parts)}",
        ])
    
    # Only show tone if set
    if state_map.tone:
        tone_str = state_map.tone.value if hasattr(state_map.tone, 'value') else state_map.tone
        lines.append(f"💬 **Tone**: {tone_str}")
    
    lines.append(f"🔄 **Turns**: {state_map.memory.turn_count}")
    
    return "\n".join(lines)


def get_extraction_few_shot_examples() -> List[Dict[str, str]]:
    """
    Get few-shot examples for extraction prompting.
    
    Returns:
        List of example input-output pairs
    """
    return [
        {
            "input": "I'm building a REST API with Python and FastAPI",
            "output": """```json
{
  "extracted_state": {
    "tech": {
      "language": "python",
      "frameworks": ["FastAPI"]
    },
    "project": {
      "project_type": "API"
    }
  },
  "confidence": 0.9,
  "raw_intent": "User is building a REST API using Python with FastAPI framework"
}
```"""
        },
        {
            "input": "Use snake_case naming and Google-style docstrings",
            "output": """```json
{
  "extracted_state": {
    "style": {
      "naming_convention": "snake_case",
      "docstring_style": "google"
    }
  },
  "confidence": 0.95,
  "raw_intent": "User is specifying code style preferences for naming and documentation"
}
```"""
        },
        {
            "input": "Make it more casual",
            "output": """```json
{
  "extracted_state": {
    "tone": "casual"
  },
  "confidence": 0.85,
  "raw_intent": "User wants to change the communication tone to be more casual"
}
```"""
        },
        {
            "input": "Add a constraint that we must never use global variables",
            "output": """```json
{
  "extracted_state": {
    "tech": {
      "constraints": ["No global variables allowed"]
    }
  },
  "confidence": 0.9,
  "raw_intent": "User is adding a hard technical constraint to avoid global variables"
}
```"""
        }
    ]


# Export all
__all__ = [
    'EXTRACTION_SYSTEM_PROMPT',
    'CONFLICT_DETECTION_SYSTEM_PROMPT',
    'GENERATION_SYSTEM_PROMPT_TEMPLATE',
    'build_extraction_prompt',
    'build_conflict_detection_prompt',
    'build_augmented_prompt',
    'build_clarification_prompt',
    'build_reflexion_prompt',
    'format_state_summary',
    'get_extraction_few_shot_examples',
]