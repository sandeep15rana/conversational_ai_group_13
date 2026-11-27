import truststore
from ddgs import DDGS

from .exceptions import ResearchAgentError
from .schemas import (
    ResearchRequest,
    ResearchResponse,
    ResearchSource,
)

# Use Windows trusted certificates.
truststore.inject_into_ssl()


class DuckDuckGoResearchTool:
    """Research tool using DDGS web search."""

    def search(
        self,
        request: ResearchRequest,
    ) -> ResearchResponse:

        try:
            results = list(
                DDGS().text(
                    request.query,
                    max_results=request.max_results,
                )
            )

            if not results:
                raise ResearchAgentError(
                    f"No search results found for: {request.query}"
                )

            sources = []

            for item in results:
                url = item.get("href")

                if not url:
                    continue

                sources.append(
                    ResearchSource(
                        title=item.get(
                            "title",
                            "Untitled source",
                        ),
                        url=url,
                        content=item.get("body", ""),
                        score=None,
                    )
                )

            return ResearchResponse(
                query=request.query,
                answer=None,
                sources=sources,
                provider="duckduckgo",
                success=True,
            )

        except ResearchAgentError:
            raise

        except Exception as exc:
            raise ResearchAgentError(
                f"DuckDuckGo search failed: {exc}"
            ) from exc