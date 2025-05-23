import streamlit as st
import pandas as pd
import cv2
import tempfile
import os
from deepface import DeepFace
import altair as alt
from datetime import timedelta

st.set_page_config(page_title="顔表情 感情分析ダッシュボード v4", layout="wide")
st.title("😶‍🌫️ 顔表情 感情分析ダッシュボード v4")
st.caption("AIが抽出した感情データをもとに、傾向やインサイトを可視化・要約します。")

uploaded_video = st.file_uploader("🎥 感情データを含む動画をアップロード", type=["mp4", "mov"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    frame_rate = 3
    results = []

    st.info("🔍 動画から顔を検出して感情を解析中...（3秒ごと）")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(1)) % (frame_rate * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
                timestamp = str(timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))
                results.append({"time": timestamp, "emotion": analysis['dominant_emotion']})
            except:
                continue

    cap.release()
    df = pd.DataFrame(results)

    st.success("✅ 感情抽出が完了しました")
    st.dataframe(df, use_container_width=True)

    st.markdown("### 📊 感情の出現傾向")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('emotion:N', title='感情'),
        y=alt.Y('count():Q', title='出現数'),
        color='emotion:N'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 結果CSVをダウンロード", csv, "emotion_results.csv", "text/csv")

    st.markdown("---")
    st.subheader("📝 感情要約コメント")
    summary = df['emotion'].value_counts().idxmax()
    st.write(f"💡 この動画では **『{summary}』** が最も多く観測されました。コンテンツの印象や空気感の分析に役立ちます。")
