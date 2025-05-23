import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="顔表情 感情分析ダッシュボード v4", layout="wide")
st.title("😶‍🌫️ 顔表情 感情分析ダッシュボード v4")
st.caption("AIが抽出した感情データをもとに、傾向やインサイトを可視化・要約します。")

# ファイルアップローダーを2つ用意
col1, col2 = st.columns(2)
with col1:
    uploaded_video = st.file_uploader("🎥 動画ファイル（任意・表示用）", type=["mp4", "mov"])
with col2:
    uploaded_csv = st.file_uploader("📊 感情分析CSVファイル（必須）", type=["csv"])

if uploaded_csv:
    # CSVデータの読み込み
    df = pd.read_csv(uploaded_csv)
    
    # 基本的な統計と可視化
    st.success("✅ 感情データの読み込みが完了しました")
    
    # データフレームの表示
    st.subheader("📋 分析データ")
    st.dataframe(df, use_container_width=True)
    
    # タブでコンテンツを整理
    tab1, tab2, tab3 = st.tabs(["感情分布", "タイムライン分析", "インサイト"])
    
    with tab1:
        st.markdown("### 📊 感情の出現傾向")
        if 'emotion' in df.columns:
            # 感情の棒グラフ
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('emotion:N', title='感情', sort='-y'),
                y=alt.Y('count():Q', title='出現数'),
                color=alt.Color('emotion:N', scale=alt.Scale(scheme='category10'))
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)
            
            # 円グラフ
            fig, ax = plt.subplots(figsize=(10, 6))
            emotion_counts = df['emotion'].value_counts()
            ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            ax.axis('equal')  # 円を維持
            st.pyplot(fig)
    
    with tab2:
        st.markdown("### ⏱️ 時間軸での感情変化")
        if 'time' in df.columns and 'emotion' in df.columns:
            # 時系列での感情変化
            # 感情ごとに数値を割り当て
            emotions_list = df['emotion'].unique()
            emotion_map = {e: i for i, e in enumerate(emotions_list)}
            df['emotion_value'] = df['emotion'].map(emotion_map)
            
            # 時系列グラフ
            timeline_chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('time:O', title='時間'),
                y=alt.Y('emotion_value:Q', title='感情', 
                       scale=alt.Scale(domain=[min(emotion_map.values()), max(emotion_map.values())]),
                       axis=alt.Axis(tickCount=len(emotions_list), 
                                    tickValues=list(emotion_map.values()), 
                                    tickLabels=list(emotion_map.keys()))),
                tooltip=['time', 'emotion']
            ).properties(
                width=800,
                height=400,
                title='時間経過による感情の変化'
            )
            st.altair_chart(timeline_chart, use_container_width=True)
            
            # 時間帯ごとの感情分布ヒートマップ
            if len(df) > 10:  # 十分なデータポイントがある場合
                # 時間を5等分
                df['time_segment'] = pd.qcut(range(len(df)), 5, labels=['開始', '序盤', '中盤', '終盤', '終了'])
                
                # クロス集計
                heatmap_data = pd.crosstab(df['time_segment'], df['emotion'])
                
                # ヒートマップデータの準備
                heatmap_df = heatmap_data.reset_index().melt(id_vars=['time_segment'], 
                                                            value_vars=heatmap_data.columns,
                                                            var_name='emotion', value_name='count')
                
                # ヒートマップ
                heatmap = alt.Chart(heatmap_df).mark_rect().encode(
                    x=alt.X('time_segment:O', title='動画セグメント'),
                    y=alt.Y('emotion:N', title='感情'),
                    color=alt.Color('count:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['time_segment', 'emotion', 'count']
                ).properties(
                    width=600,
                    height=400,
                    title='動画セグメントごとの感情分布'
                )
                st.altair_chart(heatmap, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 感情分析インサイト")
        
        # 最も多い感情
        most_common = df['emotion'].value_counts().idxmax()
        most_common_pct = df['emotion'].value_counts(normalize=True)[most_common] * 100
        
        # 感情の多様性
        emotion_diversity = len(df['emotion'].unique())
        total_emotions = len(df)
        
        # 感情の変化回数
        emotion_changes = sum(df['emotion'].shift() != df['emotion']) 
        change_rate = emotion_changes / (len(df) - 1) if len(df) > 1 else 0
        
        # インサイトの表示
        st.info(f"📌 この動画では「**{most_common}**」が支配的な感情で、全体の**{most_common_pct:.1f}%**を占めています。")
        
        if most_common_pct > 70:
            st.write("感情表現が非常に一貫しています。単調なメッセージや一定のトーンが維持されています。")
        elif most_common_pct > 50:
            st.write("比較的一貫した感情表現が見られますが、いくつかの変化点もあります。")
        else:
            st.write("感情表現に多様性があり、様々な感情が表示されています。")
        
        st.write(f"- 検出された異なる感情の種類: **{emotion_diversity}種類**")
        st.write(f"- 感情の変化回数: **{emotion_changes}回** (変化率: **{change_rate*100:.1f}%**)")
        
        if change_rate > 0.5:
            st.write("感情の変化が頻繁に起こっており、表情が豊かである可能性があります。")
        elif change_rate > 0.3:
            st.write("適度な感情の変化があり、表情のバリエーションがあります。")
        else:
            st.write("感情の変化が少なく、一貫した表情が続いています。")
        
        # 特徴的なセグメントがあれば表示
        if 'time_segment' in df.columns:
            segment_emotions = df.groupby('time_segment')['emotion'].agg(lambda x: x.value_counts().idxmax())
            if len(segment_emotions.unique()) > 1:
                st.write("### 動画セグメントごとの特徴:")
                for segment, emotion in segment_emotions.items():
                    st.write(f"- **{segment}**: 主に「**{emotion}**」の感情が表示されています")
    
    # CSVダウンロードボタン
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 結果CSVをダウンロード", csv, "emotion_results.csv", "text/csv")
    
    # 動画の表示（アップロードされている場合）
    if uploaded_video:
        st.markdown("---")
        st.subheader("📹 アップロードされた動画")
        st.video(uploaded_video)

else:
    if not uploaded_csv:
        st.warning("⚠️ 感情分析CSVファイルをアップロードしてください。")
        
        # 使い方の説明
        st.markdown("""
        ### 📝 使い方
        1. ローカル環境で動画ファイルの感情分析を実行
        2. 生成されたCSVファイルをこのアプリにアップロード
        3. 分析結果とインサイトを確認
        
        ### 📊 CSVファイル形式
        以下の列を含むCSVファイルをアップロードしてください：
        - `time`: 時間情報（例: "00:00:03"）
        - `emotion`: 検出された感情（例: "happy", "sad", "neutral"など）
        """)
        
        # サンプルデータの表示
        st.markdown("### 📋 サンプルデータ形式")
        sample_data = {
            'time': ["00:00:03", "00:00:06", "00:00:09", "00:00:12", "00:00:15"],
            'emotion': ["neutral", "happy", "happy", "surprised", "neutral"]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
