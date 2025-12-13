"""Streamlit interface for Math Tutor Agent."""

import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from latex_renderer import render_markdown_with_latex

# Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/chat"
HEALTH_ENDPOINT = f"{API_URL}/health"


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except:
        return False


def send_message(message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Send a message to the API."""
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={
                "message": message,
                "conversation_history": conversation_history
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def display_plot(plot_base64: str, width: int = 600):
    """Display a base64-encoded plot image."""
    try:
        image_data = base64.b64decode(plot_base64)
        image = Image.open(BytesIO(image_data))
        st.image(image, width=width)
    except Exception as e:
        st.error(f"Error displaying plot: {e}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Math Tutor Agent",
        page_icon="📚",
        layout="wide"
    )
    
    st.header("📚 Репетитор по математике")
    st.caption("AI-репетитор, который объясняет математику простым языком")
    
    # Check API health
    if not check_api_health():
        st.error(
            "⚠️ Cannot connect to the API. Please make sure the FastAPI server is running:\n"
            "```bash\ncd app/backend\npython -m uvicorn main:app --reload\n```"
        )
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_markdown_with_latex(message["content"])
            
            # Display plots if any
            if "plots" in message and message["plots"]:
                for plot in message["plots"]:
                    display_plot(plot, width=500)
    
    # Chat input
    if prompt := st.chat_input("Задайте вопрос по математике или попросите объяснить тему..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            render_markdown_with_latex(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = send_message(
                    prompt,
                    st.session_state.conversation_history[:-1]  # Exclude current message
                )
            
            if "error" in response:
                st.error(response["error"])
            else:
                # Display response with LaTeX support
                render_markdown_with_latex(response["response"])
                
                # Display sources if any (collapsed by default)
                if response.get("sources"):
                    st.markdown("---")
                    with st.expander("📚 Источники из учебников", expanded=False):
                        for i, source in enumerate(response["sources"], 1):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Format filename (remove extension and clean up)
                                filename = source.get('filename', 'Unknown')
                                if filename.endswith(('.djvu', '.pdf', '.doc', '.docx')):
                                    filename = filename.rsplit('.', 1)[0]
                                st.markdown(f"**{i}. {filename}**")
                            with col2:
                                level = source.get('level', 'unknown')
                                level_labels = {
                                    'elementary': 'Начальная школа',
                                    'middle_school': 'Средняя школа',
                                    'high_school': 'Старшая школа',
                                    'university': 'Университет'
                                }
                                level_label = level_labels.get(level, level)
                                st.caption(f"🎓 {level_label}")
                            
                            if source.get("text_preview"):
                                with st.container():
                                    st.markdown(f"*{source['text_preview']}*")
                            
                            if i < len(response["sources"]):
                                st.divider()
                
                # Display code execution results if any
                if response.get("code_executed"):
                    st.info("💻 Код был выполнен для решения задачи")
                
                if response.get("code_result"):
                    code_result = response["code_result"]
                    if code_result.get("plots"):
                        st.caption("📊 Графики:")
                        for plot in code_result["plots"]:
                            display_plot(plot, width=500)
                
                # Add assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": response["response"]
                }
                
                if response.get("code_result") and response["code_result"].get("plots"):
                    assistant_message["plots"] = response["code_result"]["plots"]
                
                st.session_state.messages.append(assistant_message)
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response["response"]
                })
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ О приложении")
        st.markdown("""
        AI-репетитор, который объясняет математику простым языком, используя материалы из учебников.
        
        **Источники** из учебников отображаются под каждым ответом.
        """)
        
        st.header("💡 Примеры вопросов")
        st.markdown("""
        - "Объясни, что такое производная функции"
        - "Как решать квадратные уравнения? Покажи на примере"
        - "В чём смысл интеграла? Объясни простыми словами"
        """)
        
        if st.button("🗑️ Очистить историю"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()


if __name__ == "__main__":
    main()

