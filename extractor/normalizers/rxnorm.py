# extractor/normalizers/rxnorm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from rapidfuzz import process, fuzz
import csv
import os


@dataclass
class RxNormNormalizer:
    # name -> (rxcui, canonical_generic)
    name2canon: Dict[str, Tuple[str, str]]
    choices: list

    @classmethod
    def from_tsv(cls, path: str) -> "RxNormNormalizer":
        """
        TSV columns (no header):
            name<TAB>rxcui<TAB>generic
        Example lines:
            ibuprofen\t5640\tibuprofen
            advil\t5640\tibuprofen
            motrin\t5640\tibuprofen
        You can add as many alias rows as you like.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"RxNorm TSV not found: {path}")
        name2canon: Dict[str, Tuple[str, str]] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if not row or len(row) < 3:
                    continue
                name, rxcui, generic = row[0].strip(), row[1].strip(), row[2].strip()
                if not name:
                    continue
                name2canon[name.lower()] = (rxcui, generic)
        return cls(name2canon=name2canon, choices=list(name2canon.keys()))

    def normalize(self, surface: str, score_cutoff: int = 92) -> Tuple[Optional[str], Optional[str], int]:
        """
        Return (rxcui, generic, score) for the best match or (None, None, 0).
        """
        q = (surface or "").lower().strip()
        if not q:
            return None, None, 0
        match = process.extractOne(q, self.choices, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
        if not match:
            return None, None, 0
        key, score, _ = match  # key is the matched alias
        rxcui, generic = self.name2canon.get(key, (None, None))
        return rxcui, generic, int(score)