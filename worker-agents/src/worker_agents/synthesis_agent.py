import json
from typing import Any

from .exceptions import SynthesisAgentError
from .schemas import SynthesisRequest, SynthesisResponse


SYSTEM_PROMPT = """
You are a synthesis agent in a multi-agent AI system.

Use only the supplied research findings,
analytical findings, and memory context.

Rules:
1. Answer the original question clearly.
2. Produce structured Markdown.
3. Do not invent facts.
4. Preserve available source URLs.
5. Mention limitations when evidence is insufficient.
"""


class SynthesisAgent:
    name = "synthesis"

    def __init__(self, model: Any) -> None:
        self.model = model

    def run(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResponse:

        try:
            payload = request.model_dump(mode="json")

            prompt = (
                SYSTEM_PROMPT
                + "\n\nOriginal query and findings:\n"
                + json.dumps(payload, indent=2)
            )

            response = self.model.invoke(prompt)

            content = getattr(
                response,
                "content",
                str(response),
            )

            citations = []

            if request.research_findings:
                citations = [
                    str(source.url)
                    for source
                    in request.research_findings.sources
                ]

            return SynthesisResponse(
                markdown=content,
                citations=citations,
                success=True,
            )

        except Exception as exc:
            raise SynthesisAgentError(
                f"Synthesis failed: {exc}"
            ) from exc