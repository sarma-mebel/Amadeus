"""Free, keyless scientific-literature search clients (PubMed + arXiv).

Both APIs are public and require no key, so these are $0 tools. Structured
results (title/authors/venue/link) — a smaller prompt-injection surface than
arbitrary web pages.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import httpx

LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0


class PubMedClient:
    """NCBI E-utilities client (esearch -> esummary). No API key required."""

    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    _EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def search(self, query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            r = self._client.get(
                self._ESEARCH,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "date",
                },
            )
            r.raise_for_status()
            ids = (r.json().get("esearchresult", {}) or {}).get("idlist", [])
            if not ids:
                return []
            s = self._client.get(
                self._ESUMMARY,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                },
            )
            s.raise_for_status()
            result = s.json().get("result", {}) or {}
            out: List[Dict[str, str]] = []
            for pmid in result.get("uids", []):
                item = result.get(pmid, {})
                names = [a.get("name", "") for a in item.get("authors", [])]
                authors = ", ".join(names[:4]) + (
                    f" (+{len(names)-4} more)" if len(names) > 4 else ""
                )
                out.append(
                    {
                        "pmid": str(pmid),
                        "title": str(item.get("title", "")),
                        "authors": authors,
                        "venue": str(
                            item.get("fulljournalname") or item.get("source", "")
                        ),
                        "date": str(item.get("pubdate", "")),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
                )
            return out
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("PubMed search failed: %s", exc)
            return []

    def fetch(self, title_or_pmid: str) -> Optional[Dict[str, str]]:
        """Fetch ONE paper's full record — complete authors + ABSTRACT — by
        title keywords or a PMID. Grounds detail questions in real data."""
        try:
            q = title_or_pmid.strip()
            pmid = q if q.isdigit() else None
            if pmid is None:
                r = self._client.get(
                    self._ESEARCH,
                    params={"db": "pubmed", "term": q, "retmode": "json", "retmax": 1},
                )
                r.raise_for_status()
                ids = (r.json().get("esearchresult", {}) or {}).get("idlist", [])
                if not ids:
                    return None
                pmid = ids[0]
            # efetch gives the abstract (esummary does not).
            f = self._client.get(
                self._EFETCH,
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )
            f.raise_for_status()
            root = ET.fromstring(f.text)
            art = root.find(".//Article")
            if art is None:
                return None
            title = (
                "".join(art.find(".//ArticleTitle").itertext())
                if art.find(".//ArticleTitle") is not None
                else ""
            )
            abstract = " ".join(
                "".join(node.itertext()) for node in art.findall(".//AbstractText")
            ).strip()
            authors = []
            for a in art.findall(".//Author"):
                last = a.findtext("LastName") or ""
                init = a.findtext("Initials") or ""
                if last:
                    authors.append(f"{last} {init}".strip())
            journal = art.findtext(".//Journal/Title") or ""
            return {
                "pmid": str(pmid),
                "title": title.strip(),
                "authors": ", ".join(authors),
                "abstract": abstract or "(no abstract available)",
                "venue": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("PubMed fetch failed: %s", exc)
            return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass


class ArxivClient:
    """arXiv Atom API client. No API key required."""

    _URL = "https://export.arxiv.org/api/query"
    _ATOM = "{http://www.w3.org/2005/Atom}"

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def _parse_entry(self, entry, *, full: bool) -> Dict[str, str]:
        """Parse one Atom entry. full=True keeps the complete author list and
        untruncated abstract (for fetch); full=False trims for list views."""
        title = " ".join((entry.findtext(f"{self._ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{self._ATOM}summary") or "").split())
        names = [
            (a.findtext(f"{self._ATOM}name") or "").strip()
            for a in entry.findall(f"{self._ATOM}author")
        ]
        link = (entry.findtext(f"{self._ATOM}id") or "").strip()
        date = (entry.findtext(f"{self._ATOM}published") or "")[:10]
        if full:
            authors = ", ".join(names)
            abstract = summary
        else:
            authors = ", ".join(names[:4]) + (
                f" (+{len(names)-4} more)" if len(names) > 4 else ""
            )
            abstract = summary[:500] + "…" if len(summary) > 500 else summary
        return {
            "title": title,
            "authors": authors,
            "summary": abstract,
            "url": link,
            "date": date,
            "num_authors": str(len(names)),
        }

    def search(self, query: str, *, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            # Sort by relevance, not submittedDate: with a loose all:<query>
            # match, date-descending returns the newest papers matching ANY
            # term (a common word like "geometry" then dominates, so every
            # query yields the same newest-overall junk). Relevance ranks
            # genuinely on-topic papers first — recent strong matches still
            # surface, and the per-result date is in the output either way.
            r = self._client.get(
                self._URL,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            r.raise_for_status()
            root = ET.fromstring(r.text)
            return [
                self._parse_entry(e, full=False)
                for e in root.findall(f"{self._ATOM}entry")
            ]
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("arXiv search failed: %s", exc)
            return []

    def fetch(self, title_or_id: str) -> Optional[Dict[str, str]]:
        """Fetch ONE paper's full record (complete authors + full abstract) by
        title keywords or an arXiv id. Grounds 'who wrote it / what are the
        methods' in real data instead of the model's memory."""
        try:
            q = title_or_id.strip()
            # An arXiv id (e.g. 2607.07708) -> id_list; otherwise title search.
            params = (
                {"id_list": q, "max_results": 1}
                if q.replace(".", "").replace("v", "").isdigit()
                else {
                    "search_query": f"ti:{q}",
                    "start": 0,
                    "max_results": 1,
                    "sortBy": "relevance",
                }
            )
            r = self._client.get(self._URL, params=params)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            entry = root.find(f"{self._ATOM}entry")
            return self._parse_entry(entry, full=True) if entry is not None else None
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("arXiv fetch failed: %s", exc)
            return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover
            pass
