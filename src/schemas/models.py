"""
schemas/models.py - Phase 2
Defines the exact structure we expect from the LLM.

Analogy: These are the government form templates.
Each class defines which fields are required,
what type each field must be, and what values are valid.

Pydantic automatically rejects output that does not match.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ── Schema 1: Text Sentiment ───────────────────────────────────────────────
class SentimentLabel(str, Enum):
    """
    Enum restricts the value to exactly these three options.
    The LLM cannot return "MIXED" or "neutral" (wrong case) —
    only "positive", "negative", or "neutral" exactly.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


class SentimentResult(BaseModel):
    """
    Schema for sentiment analysis output.

    Field(...) means the field is REQUIRED — model must provide it.
    Field(default=...) means it is optional with a default value.
    description= tells the LLM what the field means.
    """
    sentiment:   SentimentLabel = Field(..., description="The sentiment of the text")
    confidence:  float          = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning:   str            = Field(..., description="One sentence explaining the sentiment")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v):
        """
        Validators run automatically when Pydantic parses the output.
        If confidence is outside 0-1, Pydantic raises a ValidationError.
        The retry mechanism then asks the LLM to try again.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return round(v, 2)


# ── Schema 2: Information Extraction ──────────────────────────────────────
class ExtractedInfo(BaseModel):
    """
    Schema for extracting structured information from messy text.
    This is the most common real-world use case for fine-tuning.

    Example input:  "John Smith, 34, lives in Dubai and works as an engineer"
    Example output: {"name": "John Smith", "age": 34, "city": "Dubai", "job": "engineer"}
    """
    name:     Optional[str]   = Field(None, description="Person's full name if mentioned")
    age:      Optional[int]   = Field(None, description="Person's age as integer if mentioned")
    city:     Optional[str]   = Field(None, description="City of residence if mentioned")
    job:      Optional[str]   = Field(None, description="Job title or profession if mentioned")
    summary:  str             = Field(...,  description="One sentence summary of the text")

    @field_validator("age")
    @classmethod
    def age_must_be_realistic(cls, v):
        if v is not None and not 0 <= v <= 150:
            raise ValueError(f"Age {v} is not realistic")
        return v


# ── Schema 3: Task Classification ─────────────────────────────────────────
class TaskType(str, Enum):
    QUESTION    = "question"
    COMMAND     = "command"
    STATEMENT   = "statement"
    GREETING    = "greeting"


class ClassifiedTask(BaseModel):
    """
    Schema for classifying what type of input a user sent.
    Useful in AI assistants to route inputs to the right handler.
    """
    task_type:   TaskType = Field(..., description="The type of input")
    intent:      str      = Field(..., description="What the user wants in 5 words or less")
    requires_ai: bool     = Field(..., description="Whether this needs an AI response")


# ── Schema 4: Code Review ──────────────────────────────────────────────────
class CodeReviewResult(BaseModel):
    """
    Schema for structured code review output.
    Shows how schemas can enforce nested structure.
    """
    has_bugs:        bool       = Field(..., description="Whether bugs were found")
    bug_description: Optional[str] = Field(None, description="Description of bugs if found")
    quality_score:   int        = Field(..., description="Code quality score 1-10")
    suggestions:     list[str]  = Field(..., description="List of improvement suggestions")

    @field_validator("quality_score")
    @classmethod
    def score_must_be_valid(cls, v):
        if not 1 <= v <= 10:
            raise ValueError(f"Quality score must be 1-10, got {v}")
        return v