"""Build external MAST searches from catalog metadata, without network requests."""

from __future__ import annotations

import json
import math
import re
from typing import Sequence
from urllib.parse import quote

MAST_PORTAL = "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"
SYSTEM_TARGET_EXAMPLES = 6
_FIELD_POSITION = re.compile(r".+ \((\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)")


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _portal_url(query: dict) -> str:
    return f"{MAST_PORTAL}?searchQuery={quote(json.dumps(query, separators=(',', ':')), safe='')}"


def mast_search_url(
    missions: Sequence[object],
    *,
    obs_id: object = None,
    target_names: Sequence[object] = (),
    ra: object = None,
    dec: object = None,
) -> str | None:
    """Link to a retained dataset or raw target names, never a canonical system label.

    The ETL appends coordinates to generic field names (e.g. ``ANY (53.2, -27.8)``).
    Those labels do not exist in MAST: use a 0.1-degree cone around the first such
    field instead. Cone results can include other missions and nearby targets.
    """
    collections = list(dict.fromkeys(name for value in missions if (name := _text(value))))
    identifier = _text(obs_id)
    names = list(dict.fromkeys(name for value in target_names if (name := _text(value))))
    archive_names = [name for name in names if not _FIELD_POSITION.fullmatch(name)]
    if identifier and collections:
        field, values = "obs_id", [identifier]
    elif archive_names and collections:
        field, values = "target_name", archive_names
    else:
        if names and (position := _FIELD_POSITION.fullmatch(names[0])):
            ra, dec = position.groups()
        try:
            lon, lat = float(ra), float(dec)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lon) and math.isfinite(lat) and 0 <= lon <= 360 and -90 <= lat <= 90):
            return None
        return _portal_url({"service": "CAOMDB", "inputText": f"{lon % 360:.6f} {lat:.6f} r=0.1d"})

    filters = []
    for name, entries in (("obs_collection", collections), (field, values)):
        label = ", ".join(entries)
        filters.append(
            {
                "paramName": name,
                "niceName": name,
                "values": entries,
                "valString": label,
                "displayString": label,
                "isDate": False,
                "facetType": "discrete",
            }
        )
    return _portal_url(
        {
            "service": "CAOMFILTERED",
            "inputText": filters,
            "paramsService": "Mast.Caom.Filtered",
            "title": "MAST: " + (identifier or ", ".join(archive_names)),
            "columns": "*",
            "columnsConfig": "Mast.Caom.Cone",
        }
    )
