import streamlit as st
import gc
import nncf
import openvino as ov
from pathlib import Path
import requests

# Helper function to download files if they don't exist
def download_file(url, file_path):
    if not file_path.exists():
        response = requests.get(url)
        file_path.write_text(response.text)
        st.success(f"{file_path.name} downloaded successfully!")

# Paths for helper files
helper_file = Path("ov_nano_llava_helper.py")
cmd_helper_file = Path("cmd_helper.py")

# Download necessary files
st.write("Checking for required files...")
download_file(
    f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/nano-llava-multimodal-chatbot/{helper_file.name}",
    helper_file,
)
download_file(
    f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/{cmd_helper_file.name}",
    cmd_helper_file,
)

# Import downloaded modules
from cmd_helper import optimum_cli
from ov_nano_llava_helper import converted_model_exists, copy_model_files

# Streamlit UI
st.title("Nano LLAVA Multimodal Chatbot with OpenVINO")
st.write("This app uses OpenVINO for optimizing multimodal chatbot models.")

# Check if a converted model exists
if st.button("Check Converted Model"):
    if Path("cmd_helper.py").exists():
    
        st.success("Converted model exists!")
    else:
        st.warning("Converted model does not exist!")

# Placeholder for additional functionality
st.write("Add more interactive features as needed, such as uploading files or running the chatbot.")
