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

/* full-bleed column: drop Streamlit's gutters (our blocks pad themselves), float the
   collapse button over the top instead of reserving a 76px band, and stretch the
   content to full height so the footer can be pinned to the bottom.
   Markdown containers carry a -16px bottom margin meant to cancel a trailing <p>
   margin; our blocks are divs, so it would drag each next element up over them. */
[data-testid="stSidebarContent"] {{
  padding: 0 !important; scrollbar-gutter: auto; display: flex; flex-direction: column;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {{ width: 100% !important; }}
[data-testid="stSidebarHeader"] {{
  position: absolute; top: 0; right: 0; z-index: 2;
  height: auto; margin: 0; padding: 12px 10px;
}}
[data-testid="stSidebarUserContent"] {{ padding: 0 !important; flex: 1; display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] > div {{ flex: 1; display: flex; flex-direction: column; }}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{ gap: 0; flex: 1; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sb-info) {{ margin-top: auto; }}

/* header block */
.sb-head {{ padding: 26px 20px 18px; border-bottom: 1px solid rgba(255,255,255,.1); }}
.sb-title {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 21px; letter-spacing: .05em; text-transform: uppercase; line-height: 1.1;
}}
.sb-sub {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 500;
  font-size: 12.5px; letter-spacing: .16em; text-transform: uppercase;
  color: #8a97a5 !important; margin-top: 7px;
}}
.sb-seclabel {{
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600;
  font-size: 11px; letter-spacing: .18em; text-transform: uppercase;
  color: #6f7d8c !important; padding: 16px 20px 7px;
}}

/* radio -> full-bleed nav rows (Streamlit 1.63 renders the radio with React Aria:
   label[data-testid=stRadioOption] > span(hidden input) + div > div > [marker, text]) */
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{ width: 100%; gap: 0; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {{
  width: 100%; margin: 0; padding: 10px 20px 10px 17px; cursor: pointer;
  border-left: 3px solid transparent; transition: background .12s;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,.05); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{
  background: {ACCENT_700}; border-left-color: {ACCENT};
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) {{
  display: none !important;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div,
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div,
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {{
  flex: 1; width: 100%; min-width: 0;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{
  display: flex; align-items: baseline; margin: 0;
  font-family: 'Barlow Condensed', sans-serif !important; font-weight: 600;
  font-size: 15px !important; letter-spacing: .09em; text-transform: uppercase; line-height: 1.25;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p strong {{
  margin-left: auto; padding-left: 12px; font-weight: 600; font-size: 12.5px;
  letter-spacing: .04em; font-variant-numeric: tabular-nums; color: #8a97a5 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p strong {{ color: #d6dfe8 !important; }}

/* footer info block, pinned to the bottom */
.sb-info {{ border-top: 1px solid rgba(255,255,255,.1); padding: 14px 20px 18px; }}
.sb-info .row {{
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 12px; padding: 3px 0; font-size: 12.5px;
}}
.sb-info .row .k {{ color: #8a97a5 !important; }}
.sb-info .row .v {{ font-variant-numeric: tabular-nums; text-align: right; }}

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
