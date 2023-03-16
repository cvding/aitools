import streamlit as st
from time import sleep
import torch

def load_model(outputs):
    with st.spinner("Loading model..."):
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
        convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
        st.session_state['rvm'] = {'model': model, 'func': convert_video}

class Main:
    def __init__(self) -> None:
        self.tabs = [st.tabs(['抠图'])]
        outputs = {}
        st.button('加载模型', on_click=load_model, args=(outputs,))
        
        with self.tabs[0][0]:
            uploaded_file = st.file_uploader('图片文件')
            my_bar = st.progress(0)
            left_col, right_col = st.columns(2)
            with left_col:
                st.selectbox('backbone',['mobilenetv3', 'resnet50'])
                st.selectbox('output-type', ['video', 'image'])

                if uploaded_file is not None:
                    bytes_data = uploaded_file.getvalue()
                    st.image(bytes_data)
            with right_col:
                st.selectbox('device', ['cuda', 'cpu'])
                st.number_input('seq_chunk', min_value=1, value=1, format='%d')
                if uploaded_file is not None:
                    st.session_state['rvm']['convert_video']()
                    for i in range(100):
                        sleep(0.05)
                        my_bar.progress(i+1)
                    st.image(bytes_data)

            lc, rc = st.columns([0.8, 0.2])
            with lc:
                st.text_input('', placeholder='input your image list or director',label_visibility="collapsed")
            with rc:
                st.button('处理',use_container_width=True)


main = Main()