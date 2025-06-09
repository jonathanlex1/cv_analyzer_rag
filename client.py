import streamlit as st 
import requests


def main() : 

    if 'chat_disabled' not in st.session_state : 
        st.session_state.chat_disabled = False
    
    st.header('🤖 RAG Chatbot to Analyze and Match Your CV with Job Requirements')

    #upload pdf document
    with st.sidebar:
        st.title("🤖 RAG CV Chatbot")
        st.markdown("### 🎯 Let's Match Your CV to Your Dream Job!")
        st.markdown("Upload your **CV (PDF)** and chat with an AI assistant to check how well you match your desired job.")
        
        uploaded_file = st.file_uploader("📄 Upload your CV here:", type="pdf")

        if st.button('Reset') : 
            st.session_state.chat_disabled = False

        st.markdown("---")
        st.markdown("💡 *Tip: Make sure your resume is updated for best results!*")


    if not st.session_state.chat_disabled :    
        query = st.chat_input('Tell the job requirements that you want to apply')
        with st.spinner('Analyzing...') :
            if query and uploaded_file : 
                files = {'file' : (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {'prompt' : query}
                response = requests.post(' http://127.0.0.1:8000', 
                                        files=files,
                                        data=data)
                if response.ok : 
                    result = response.json()
                    st.markdown(result['answer'])
                else : 
                    st.warning('error from API')
                
                st.session_state.chat_disabled = True

if __name__ == '__main__' : 
    main()