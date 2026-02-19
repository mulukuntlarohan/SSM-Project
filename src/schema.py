from pydantic import BaseModel, Field
from typing import List, Optional

class TechnicalState(BaseModel):
    language: str = Field(default="Not Specified", description="Primary programming language.")
    framework: str = Field(default="None", description="e.g. React, FastAPI, etc.")
    constraints: List[str] = Field(default_factory=list, description="Strict technical rules (e.g. 'No loops').")

class StyleState(BaseModel):
    tone: str = Field(default="Neutral", description="Professional, Sarcastic, Concise, etc.")
    vibe: List[str] = Field(default_factory=list, description="Keywords for the aesthetic or mood.")

class GlobalState(BaseModel):
    """The Master JSON Map that will prevent Semantic Drift."""
    tech: TechnicalState
    style: StyleState
    project_goal: str = Field(default="Unknown", description="The overall objective of the session.")
    last_updated_turn: int = 0