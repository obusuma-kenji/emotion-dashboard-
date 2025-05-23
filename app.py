import streamlit as st
import pandas as pd
import cv2
import numpy as np
from datetime import datetime
import time

# ページ設定
st.set_page_config(page_title="感情分析ダッシュボード", layout="wide")

# タイトル
st.title("リアルタイム感情分析ダッシュボード")
st.markdown("OpenCVを使った顔検出と感情分析のデモです")

# サイドバー
st.sidebar.header("コントロールパネル")

# データ収集のための変数初期化
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []

# 感情のモックデータ（実際はDeepFaceが担当する部分）
emotions = ['happy', 'sad', 'angry', 'neutral', 'surprised', 'fear', 'disgust']

# 感情分析関数（モック - 実際はDeepFaceが担当する部分）
def analyze_emotion(face_img):
    # ランダムな感情と確率を返す（デモ用）
    emotion_probs = np.random.dirichlet(np.ones(len(emotions)))
    emotion_dict = dict(zip(emotions, emotion_probs))
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    return dominant_emotion, emotion_dict

# カメラからのリアルタイム分析
def realtime_analysis():
    # OpenCVのカスケード分類器をロード
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"分類器のロードに失敗しました: {e}")
        return
    
    # カメラの設定
    st.write("### リアルタイム感情分析")
    run = st.checkbox("カメラをオンにする")
    
    if run:
        FRAME_WINDOW = st.image([])
        
        # 分析結果表示用
        result_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        while run:
            # デモ用に画像をシミュレート
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            img[:] = (200, 200, 200)  # グレー背景
            
            # ランダムな位置に顔の円を描画
            center_x = np.random.randint(100, 300)
            center_y = np.random.randint(100, 200)
            cv2.circle(img, (center_x, center_y), 50, (255, 200, 200), -1)  # 顔
            
            # 分析
            dominant_emotion, emotion_dict = analyze_emotion(img)
            
            # 時間情報の追加
            timestamp = datetime.now().strftime("%H:%M:%S")
            emotion_data = {"time": timestamp}
            emotion_data.update(emotion_dict)
            
            # データ保存
            st.session_state.emotion_data.append(emotion_data)
            if len(st.session_state.emotion_data) > 100:  # 最大100ポイント保存
                st.session_state.emotion_data.pop(0)
            
            # 結果の表示
            FRAME_WINDOW.image(img, channels="BGR")
            result_placeholder.write(f"検出された主要な感情: **{dominant_emotion}** ({emotion_dict[dominant_emotion]:.2f})")
            
            # 感情グラフ
            df = pd.DataFrame(st.session_state.emotion_data)
            if len(df) > 0:
                df = df.set_index('time')
                chart_placeholder.line_chart(df)
            
            time.sleep(1)  # 1秒おきに更新

# 履歴データの表示
def display_historical_data():
    st.write("### 感情分析の履歴")
    
    if len(st.session_state.emotion_data) > 0:
        df = pd.DataFrame(st.session_state.emotion_data)
        st.dataframe(df)
        
        # 感情の分布をグラフ化
        st.write("### 感情の分布")
        emotion_df = df.drop('time', axis=1)
        mean_emotions = emotion_df.mean().sort_values(ascending=False)
        st.bar_chart(mean_emotions)
    else:
        st.info("データがありません。リアルタイム分析を実行してデータを収集してください。")

# メインアプリ
def main():
    option = st.sidebar.selectbox(
        "機能を選択してください",
        ["リアルタイム分析", "履歴データ"]
    )
    
    if option == "リアルタイム分析":
        realtime_analysis()
    elif option == "履歴データ":
        display_historical_data()

if __name__ == "__main__":
    main()
