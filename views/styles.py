import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .page-title {
            font-size: 2.35rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
            color: #132033;
        }
        .page-subtitle {
            color: #526173;
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 900px;
            margin-bottom: 1.2rem;
        }
        .info-card {
            border: 1px solid #dde3ea;
            border-radius: 8px;
            padding: 1.15rem 1.2rem;
            background: #ffffff;
            min-height: 172px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .info-card h3 {
            color: #172033;
            font-size: 1.08rem;
            margin: 0 0 0.55rem;
        }
        .info-card p {
            color: #526173;
            line-height: 1.6;
            margin: 0;
        }
        .avatar-placeholder {
            width: 92px;
            height: 92px;
            border-radius: 50%;
            margin: 0 auto 0.85rem;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #e7eef5;
            color: #2d4b6c;
            font-size: 1.55rem;
            font-weight: 700;
        }
        .profile-name {
            color: #172033;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            text-align: center;
        }
        .profile-email {
            color: #526173;
            font-size: 0.9rem;
            overflow-wrap: anywhere;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
