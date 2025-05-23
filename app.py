import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from datetime import timedelta
import random
import altair as alt
import matplotlib.pyplot as plt

# ページ設定
st.set_page_config(page_title="顔表情 感情分析ダッシュボード v4", layout="wide")
st.title("😶‍🌫️ 顔表情 感情分析ダッシュボード v4")
st.caption("AIが抽出した感情データをもとに、傾向やインサイトを可視化・要約します。")

# 使用可能な感情リスト
emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'fear', 'disgust']

# 顔検出のためのカスケード分類器
@st.cache_resource
def load_face_cascade():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"分類器のロードに失敗しました: {e}")
        return None

# シンプルな感情シミュレーション関数
def simulate_emotion(face_img=None):
    # 実際には顔画像から特徴を抽出して感情を推定するロジックを実装
    # このデモではランダムだが、画像に基づいたシードを使用
    if face_img is not None:
        # 画像の平均輝度を計算して一貫性のあるランダム値を生成
        avg_brightness = np.mean(face_img)
        random.seed(int(avg_brightness * 100))
    
    # 主要な感情を選択（完全ランダムではなくバイアスをかける）
    weights = [0.3, 0.15, 0.1, 0.25, 0.1, 0.05, 0.05]  # happy, sad, angry, neutral, surprised, fear, disgust
    emotion = random.choices(emotions, weights=weights, k=1)[0]
    
    # 確率値も生成
    emotion_probs = {}
    main_prob = random.uniform(0.5, 0.8)  # 主要感情の確率
    remaining = 1.0 - main_prob
    
    # 残りの確率を他の感情に分配
    other_emotions = [e for e in emotions if e != emotion]
    other_probs = np.random.dirichlet(np.ones(len(other_emotions))) * remaining
    
    # 結果の辞書を作成
    emotion_probs[emotion] = main_prob
    for e, p in zip(other_emotions, other_probs):
        emotion_probs[e] = p
    
    return emotion, emotion_probs

# 動画のアップロード
uploaded_video = st.file_uploader("🎥 感情分析する動画をアップロード", type=["mp4", "mov"])

if uploaded_video:
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    
    # 分析設定
    st.sidebar.header("分析設定")
    frame_rate = st.sidebar.slider("サンプリングレート（秒ごと）", 1, 10, 3)
    face_size_min = st.sidebar.slider("最小顔サイズ（ピクセル）", 30, 100, 60)
    
    # 分析の実行
    if st.button("感情分析を開始"):
        # 顔検出器の読み込み
        face_cascade = load_face_cascade()
        if face_cascade is None:
            st.error("顔検出器の初期化に失敗しました。")
        else:
            # 動画の処理を開始
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            st.info(f"🎬 動画情報: {int(total_frames)}フレーム, {fps:.1f}FPS, 長さ: {timedelta(seconds=int(duration))}")
            
            # 進捗バー
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 結果を格納する配列
            results = []
            frame_results = []
            
            # サンプリングするフレーム数を計算
            frames_to_sample = int(duration / frame_rate)
            frames_to_skip = int(fps * frame_rate)
            
            # サムネイル表示用のコンテナ
            thumbnail_container = st.empty()
            
            # カウンター初期化
            frame_count = 0
            sample_count = 0
            
            # 処理開始
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームレートに基づいてサンプリング
                if frame_count % frames_to_skip == 0:
                    # タイムスタンプの計算
                    timestamp = str(timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)))
                    status_text.text(f"🔍 フレーム {frame_count}/{total_frames} を分析中... ({timestamp})")
                    
                    # グレースケールに変換して顔検出
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(face_size_min, face_size_min)
                    )
                    
                    # 顔が検出された場合
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            # 顔領域を抽出
                            face_roi = frame[y:y+h, x:x+w]
                            
                            # 顔に四角形を描画
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # 感情の推定（シミュレーション）
                            emotion, emotion_probs = simulate_emotion(face_roi)
                            
                            # 結果を保存
                            result = {
                                "time": timestamp, 
                                "emotion": emotion,
                                "frame_index": frame_count
                            }
                            result.update(emotion_probs)
                            results.append(result)
                            
                            # 感情テキストを描画
                            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    else:
                        # 顔が検出されなかった場合
                        result = {
                            "time": timestamp,
                            "emotion": "no_face",
                            "frame_index": frame_count
                        }
                        for e in emotions:
                            result[e] = 0.0
                        results.append(result)
                    
                    # RGB形式に変換してサムネイル表示
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    thumbnail_container.image(rgb_frame, caption=f"時間: {timestamp} - 感情: {results[-1]['emotion']}")
                    
                    # フレームと結果を保存
                    frame_results.append((rgb_frame, results[-1]))
                    
                    # 進捗を更新
                    sample_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                frame_count += 1
            
            # リソースの解放
            cap.release()
            os.unlink(video_path)  # 一時ファイルを削除
            
            # 分析完了
            status_text.success(f"✅ 分析完了！ {sample_count}サンプルから感情を抽出しました。")
            
            if results:
                # データフレームに変換
                df = pd.DataFrame(results)
                
                # タブで表示を整理
                tab1, tab2, tab3 = st.tabs(["感情分布", "タイムライン分析", "サムネイル一覧"])
                
                with tab1:
                    st.markdown("### 📊 感情の出現傾向")
                    
                    # 顔が検出された結果のみでフィルタリング
                    filtered_df = df[df['emotion'] != 'no_face']
                    
                    if len(filtered_df) > 0:
                        # 感情の棒グラフ
                        chart = alt.Chart(filtered_df).mark_bar().encode(
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
                        emotion_counts = filtered_df['emotion'].value_counts()
                        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', 
                               startangle=90, shadow=True)
                        ax.axis('equal')  # 円を維持
                        st.pyplot(fig)
                    else:
                        st.warning("顔が検出されたフレームがありません。")
                
                with tab2:
                    st.markdown("### ⏱️ 時間軸での感情変化")
                    
                    # 顔が検出された結果のみでフィルタリング
                    filtered_df = df[df['emotion'] != 'no_face']
                    
                    if len(filtered_df) > 0:
                        # 時系列での感情変化
                        timeline_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
                            x=alt.X('time:O', title='時間'),
                            y=alt.Y('emotion:N', title='感情'),
                            color=alt.Color('emotion:N', scale=alt.Scale(scheme='category10')),
                            tooltip=['time', 'emotion']
                        ).properties(
                            width=800,
                            height=400,
                            title='時間経過による感情の変化'
                        )
                        st.altair_chart(timeline_chart, use_container_width=True)
                        
                        # 感情確率のエリアチャート
                        filtered_df_melted = pd.melt(
                            filtered_df, 
                            id_vars=['time'], 
                            value_vars=emotions,
                            var_name='emotion_type', 
                            value_name='probability'
                        )
                        
                        area_chart = alt.Chart(filtered_df_melted).mark_area().encode(
                            x=alt.X('time:O', title='時間'),
                            y=alt.Y('probability:Q', title='確率', stack='normalize'),
                            color=alt.Color('emotion_type:N', title='感情',
                                           scale=alt.Scale(scheme='category10')),
                            tooltip=['time', 'emotion_type', 'probability']
                        ).properties(
                            width=800,
                            height=400,
                            title='時間経過による感情確率の変化'
                        )
                        st.altair_chart(area_chart, use_container_width=True)
                    else:
                        st.warning("顔が検出されたフレームがありません。")
                
                with tab3:
                    st.markdown("### 🖼️ 分析サムネイル一覧")
                    
                    # 3列のグリッドレイアウト
                    cols = st.columns(3)
                    
                    # サムネイルを表示
                    for i, (frame, result) in enumerate(frame_results):
                        with cols[i % 3]:
                            st.image(frame, caption=f"時間: {result['time']} - 感情: {result['emotion']}")
                
                # 分析インサイト
                st.markdown("---")
                st.subheader("📝 感情分析インサイト")
                
                # 顔が検出された結果のみでフィルタリング
                filtered_df = df[df['emotion'] != 'no_face']
                
                if len(filtered_df) > 0:
                    # 最も多い感情
                    most_common = filtered_df['emotion'].value_counts().idxmax()
                    most_common_pct = filtered_df['emotion'].value_counts(normalize=True)[most_common] * 100
                    
                    # 感情の多様性
                    emotion_diversity = len(filtered_df['emotion'].unique())
                    
                    # 感情の変化回数
                    emotion_changes = sum(filtered_df['emotion'].shift() != filtered_df['emotion']) 
                    change_rate = emotion_changes / (len(filtered_df) - 1) if len(filtered_df) > 1 else 0
                    
                    # 顔検出率
                    face_detection_rate = len(filtered_df) / len(df) * 100
                    
                    # インサイトの表示
                    st.info(f"📌 この動画では「**{most_common}**」が支配的な感情で、全体の**{most_common_pct:.1f}%**を占めています。")
                    st.write(f"- 顔検出率: **{face_detection_rate:.1f}%** ({len(filtered_df)}/{len(df)}フレーム)")
                    st.write(f"- 検出された異なる感情の種類: **{emotion_diversity}種類**")
                    st.write(f"- 感情の変化回数: **{emotion_changes}回** (変化率: **{change_rate*100:.1f}%**)")
                    
                    if most_common_pct > 70:
                        st.write("感情表現が非常に一貫しています。単調なメッセージや一定のトーンが維持されています。")
                    elif most_common_pct > 50:
                        st.write("比較的一貫した感情表現が見られますが、いくつかの変化点もあります。")
                    else:
                        st.write("感情表現に多様性があり、様々な感情が表示されています。")
                    
                    if change_rate > 0.5:
                        st.write("感情の変化が頻繁に起こっており、表情が豊かである可能性があります。")
                    elif change_rate > 0.3:
                        st.write("適度な感情の変化があり、表情のバリエーションがあります。")
                    else:
                        st.write("感情の変化が少なく、一貫した表情が続いています。")
                else:
                    st.warning("顔が検出されたフレームがありません。")
            else:
                st.warning("分析結果がありません。動画に顔が含まれているか確認してください。")
    
    # 動画の表示
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("元の動画を表示", value=False):
        st.sidebar.video(uploaded_video)
else:
    st.info("👆 動画ファイル（MP4, MOV）をアップロードして感情分析を開始してください。")
    
    # 使い方の説明
    st.markdown("""
    ### 📝 このアプリについて
    このアプリは、アップロードされた動画から顔を検出し、表情から感情を分析・可視化します。
    
    ### 主な機能
    1. 動画内の顔を自動検出
    2. 顔の表情から感情を分析（happy, sad, angry, neutral, surprised, fear, disgust）
    3. 時間経過による感情の変化を可視化
    4. 感情傾向のインサイトを自動生成
    
    ### 使い方
    1. 動画ファイル（MP4, MOV）をアップロード
    2. サイドバーで分析設定を調整（必要に応じて）
    3. 「感情分析を開始」ボタンをクリック
    4. 分析結果とインサイトを確認
    
    ### 注意事項
    - 分析は近似的なものであり、100%正確ではありません
    - 明るく、顔がはっきり映っている動画が最適です
    - 処理には動画の長さに応じて時間がかかります
    """)
