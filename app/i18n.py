"""Minimal bilingual layer (EN/中文) for this GammaGrid fork.

A fork of an AGPL project — this file, and every `i18n.t(...)` call site it
touches, is part of the modification set that must stay published with the
source. Upstream strings are passed verbatim as the first argument so a
rebase onto upstream stays a text-level conflict, never a silent one.

The language is a per-session widget state, defaulting to Chinese for this
installation's audience; the toggle lives at the top of the sidebar.
"""

from __future__ import annotations

import streamlit as st

LANG_KEY = "gg_lang"
DEFAULT_LANG = "zh"  # this installation serves Chinese-speaking users first

_LANG_CHOICES = {"中文": "zh", "English": "en"}
_LABEL_FOR = {"zh": "中文", "en": "English"}


def lang() -> str:
    """The active language code ("zh" / "en") for this session.

    Defensive on purpose: this module is imported by the worker process too
    (collector → day_summary → weather), where no Streamlit runtime exists —
    any session_state access there raises, and the worker must keep running.
    """
    try:
        value = st.session_state.get(LANG_KEY, DEFAULT_LANG)
    except Exception:
        return DEFAULT_LANG
    return value if value in ("zh", "en") else DEFAULT_LANG


def is_zh() -> bool:
    return lang() == "zh"


def t(en: str, zh: str) -> str:
    """Pick the string for the active language. `en` stays the call-site key."""
    return zh if is_zh() else en


def toggle() -> None:
    """The sidebar language switch. Re-rendered on every rerun like any widget."""
    choice = st.radio(
        "🌐 Language / 语言",
        list(_LANG_CHOICES),
        index=0 if is_zh() else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state[LANG_KEY] = _LANG_CHOICES[choice]


def label() -> str:
    return _LABEL_FOR[lang()]
