"""Restyles Streamlit's own chrome (sidebar, widgets, buttons) to match Industry.

Call `inject()` once at the top of the app, right after `st.set_page_config`.
Streamlit's internal class names change between versions, so this deliberately
targets stable attributes (`data-testid`, element types) only.
"""

import streamlit as st

from ui import ACCENT, ACCENT_700, ACCENT_900, BG, DIVIDER, TEXT

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Barlow', system-ui, sans-serif; }}
h1, h2, h3, h4, h5 {{
  font-family: 'Barlow Condensed', sans-serif !important;
  letter-spacing: .02em; text-transform: uppercase;
}}
.stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ padding: 1rem 1.4rem 3rem; max-width: 100%; }}

/* dark steel rail */
[data-testid="stSidebar"] {{ background: {ACCENT_900}; }}
[data-testid="stSidebar"] * {{ color: {BG} !important; }}

/* sidebar header block */
.sb-title {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 19px; letter-spacing: .04em; text-transform: uppercase;
  line-height: 1.1; margin: 2px 0 2px;
}}
.sb-sub {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 500;
  font-size: 12px; letter-spacing: .16em; text-transform: uppercase;
  color: #8a97a5 !important; margin-bottom: 4px;
}}
.sb-seclabel {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
  font-size: 10px; letter-spacing: .18em; text-transform: uppercase;
  color: #6f7d8c !important; padding: 0 0 2px; margin-top: 6px;
  border-bottom: 1px solid rgba(255,255,255,.08);
}}

/* radio -> full-width nav rows with selected highlight */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 0 !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  display: flex; align-items: center; width: 100%;
  padding: 9px 10px; margin: 0; cursor: pointer;
  border-left: 3px solid transparent;
  font-family: 'Barlow Condensed', sans-serif !important; font-weight: 600 !important;
  font-size: 14px !important; letter-spacing: .1em; text-transform: uppercase;
  transition: background .12s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: rgba(255,255,255,.05);
}}
/* hide only the small round radio marker, not the text.
   BaseWeb renders it as a fixed-size box; target the first flex child that is
   NOT the label-text wrapper. The text lives in a div containing a <p>, so we
   hide the sibling div that has no <p>. */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
  min-width: 0 !important; width: 0 !important; margin: 0 !important;
  overflow: hidden !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child > div {{
  display: none !important;
}}
/* selected row highlight, keyed off BaseWeb's data-selected attribute */
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{
  background: {ACCENT}; border-left-color: #cdd8e3;
}}

/* sidebar footer info block */
.sb-info {{ margin-top: 4px; }}
.sb-info .row {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 4px 0; border-top: 1px solid rgba(255,255,255,.08);
  font-size: 11px;
}}
.sb-info .row .k {{ color: #8a97a5 !important; letter-spacing: .08em; text-transform: uppercase; font-size: 10px; }}
.sb-info .row .v {{ font-variant-numeric: tabular-nums; font-weight: 600; }}

/* square, hairline widgets */
input, textarea, select,
[data-baseweb="input"], [data-baseweb="select"] > div,
[data-baseweb="textarea"] {{
  border-radius: 0 !important; border-color: {DIVIDER} !important;
  background: #fff !important; font-family: 'Barlow', sans-serif !important;
}}
input:focus-visible, textarea:focus-visible {{
  outline: 2px solid {ACCENT}; outline-offset: 2px;
}}
label, [data-testid="stWidgetLabel"] p {{
  font-size: 11px !important; letter-spacing: .1em; text-transform: uppercase;
  color: #6b6f73;
}}

/* buttons: solid accent primary, square corners */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
  border-radius: 0; border: 1px solid {ACCENT}; background: {ACCENT}; color: #fff;
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; padding: .35rem 1rem;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  background: {ACCENT_700}; border-color: {ACCENT_700}; color: #fff;
}}
[data-testid="stForm"] {{ border: 1px solid {DIVIDER}; border-radius: 0; padding: 14px; }}

/* trim Streamlit chrome that reads as "demo app" */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
iframe {{ border: 0; display: block; }}
[data-testid="stVerticalBlock"] {{ gap: .75rem; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
