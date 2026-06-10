import streamlit as st

from config import TEAM_MEMBERS
from views.styles import render_page_header


def get_initials(name):
    return "".join(part[0] for part in name.split()[:2]).upper()


def render_member_card(member):
    with st.container(border=True):
        if member["photo"]:
            st.image(member["photo"], width=92)
        else:
            st.markdown(
                f'<div class="avatar-placeholder">{get_initials(member["name"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="profile-name">{member["name"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="profile-email">{member["email"]}</div>',
            unsafe_allow_html=True,
        )


def render_contact():
    render_page_header(
        "Contact",
        "Project member information can be added here, including photos, names, and email addresses.",
    )

    for start in range(0, len(TEAM_MEMBERS), 3):
        cols = st.columns(3)
        for col, member in zip(cols, TEAM_MEMBERS[start : start + 3]):
            with col:
                render_member_card(member)

    st.caption("Update TEAM_MEMBERS in config.py to replace placeholder names, emails, and photos.")
