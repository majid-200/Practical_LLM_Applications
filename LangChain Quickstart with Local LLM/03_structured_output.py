from pprint import pprint
from typing import Literal

from pydantic import BaseModel, Field
from setup import model

class EngineSpecs(BaseModel):
    manufacturer: str = Field(description="The brand that built the engine")
    configuration: Literal["V8", "V10", "V12", "W16"] = Field(description="Cylinder layout")
    displacement_liters: float = Field(description="Engine size in liters")
    aspiration: Literal["Naturally Aspirated", "Turbocharged", "Supercharged"] = Field(description="Induction type")
    redline_rpm: int = Field(description="Maximum RPM")

structured_llm = model.with_structured_output(EngineSpecs)

prompt = """
The Ferrari 812 Superfast is a beast. It's got that massive 6.5L F140 GA engine up front.
It screams all the way to 8900 RPM without any turbos choking the sound.
It's pure Italian V12 magic.
"""

specs = structured_llm.invoke(prompt)
pprint(specs.model_dump(), indent=2)