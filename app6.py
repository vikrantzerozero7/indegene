import streamlit as st
from streamlit_image_select import image_select

# Custom CSS for centering the layout and equal spacing between buttons
st.markdown(
    """
    <style>
    .main {
        padding-top: 300px; /* Adjust this value to move the interface further down */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .stButton > button {
        width: 100%; /* Ensure buttons occupy full width for equal spacing */
        margin: 10px 0; /* Add vertical margin to space out buttons */
    }

    .stColumns {
        display: flex;
        justify-content: space-evenly; /* Evenly space out the columns */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import warnings
import fitz  # PyMuPDF
import base64
import re
from langchain.text_splitter import CharacterTextSplitter
import camelot
from huggingface_hub import login

# Suppress all warnings
warnings.filterwarnings("ignore")

# Option 1: Log in using huggingface-cli login
login("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")

from optimum.intel import OVModelForCausalLM
model = OVModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",load_in_8bit=True, export=True)

# Ensure pdf_d is defined with file paths
# pdf_d = ["path/to/your/pdf1.pdf"]  # Add more file paths as needed
#huggingface-cli login --token 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
pdf_d = [
"ast_sci_data_tables_sample.pdf",
   #"C:/Users/VIKRANT/onedrive/Desktop/INDEGENE/20200125041045198204Electrical Machines by Mr. S. K. Sahdev.pdf",
   # "/kaggle/input/indegene3/AI_Russell_Norvig.pdf"
] 

#from optimum.intel import OVModelForCausalLM

#from pypdf import PdfReader
pdf_data = []
meta_data = []

# Regex pattern for extracting links
url_pattern = r'(https?://[^\s]+)'

for pdf_path in pdf_d[:]:  # Loop through the first PDF, or adjust to handle all
    doc = fitz.open(pdf_path)
    full_content = ""

    for page_number in range(len(doc)):
        page = doc[page_number]
        page_content = page.get_text("text") or ""
        full_content += page_content  # Concatenate page contents

        page_images = []
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_data = base64.b64encode(image_bytes).decode("utf-8")  # Convert image to Base64
            page_images.append(image_data)

        # Extract links from the page content using regex
        page_links = re.findall(url_pattern, page_content)  # Find all URLs in the page content

        # Table extraction using Camelot
        tables = []
        camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_number + 1), flavor='lattice')  # Use 'stream' flavor for text-based tables
        if camelot_tables: 
            for table in camelot_tables:
                tables.append(table.df)  # Convert table to dictionary format

        # Add metadata
        meta_data.append({
            "page_number": page_number + 1,  # Pages are 0-indexed in fitz
            "images": f"{page_images}" if page_images else "No images found",
            "tables": tables if tables else "No tables found",  # Include tables or placeholder
            "links": page_links if page_links else "No links found" # Append the extracted links
        })
 
        if page_content.strip() == "":
            page_content = "No content available"  # Fallback for empty pages
        pdf_data.append(page_content)

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",  # Separator between chunks
    chunk_size=20000,  # Size of each chunk
    chunk_overlap=1000,  # Overlap between chunks
    length_function=len,  # Function to calculate the length of each chunk
    is_separator_regex=False,  # Don't treat separator as a regex
)

# Split the text into chunks and create documents
documents = text_splitter.create_documents(
    pdf_data, metadatas=meta_data
)

# Uncomment to inspect the first document after splitting
# print(documents[0])

def main():
    # Initialize session state for section control
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Image Selection"

    # Create three buttons side by side for navigation
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Go to Image Selection"):
            st.session_state.active_section = "Image Selection"

    with col2:
        if st.button("Go to Radio Button Selection"):
            st.session_state.active_section = "Radio Button Selection"

    with col3:
        if st.button("Go to Links List"):
            st.session_state.active_section = "Links List"

    # Display the current section based on button clicks
    if st.session_state.active_section == "Image Selection":
        st.header("Select an Image")
        img = image_select(
            "Select an Image", 
            [
                "cat1.jpeg", 
                "cat2.jpeg"
            ]
        )

        st.write("You selected:", img)

        user_input_image = st.text_input("Enter your input for the image:", key="user_input_image")

        if st.button("Submit Image Input"):
            # Display the selected image with reduced size
            if img:
                st.image(img, caption="Selected Image", use_column_width=False, width=200)
            
            # Display the input value
            if user_input_image:
                st.write("Your input for the image:", user_input_image)

    elif st.session_state.active_section == "Radio Button Selection":
        st.header("Select One Entry")

        radio_entries = documents[0].metadata["tables"]#["Entry 1", "Entry 2", "Entry 3"]
        table_labels = [f"Table {i+1}" for i in range(len(radio_entries))]
        entry_mapping = {f"Table {i+1}": radio_entries[i] for i in range(len(radio_entries))}

        selected_entry = st.radio(
            "Choose one:", 
            table_labels, 
            key="selected_entry", 
            index=table_labels.index(st.session_state.get("selected_entry", table_labels[0]))
        )

        if selected_entry in entry_mapping:
            st.write("Selected Entry Details:", entry_mapping[selected_entry])

        user_input_radio = st.text_input(
            "Enter your input here:", 
            placeholder="Ask a query and press Enter", 
            key="user_input_radio"
        )

        if st.button("Submit Entry Input"):
              input_prompt = f"""Context:{entry_mapping[selected_entry]}\nQuestion:{user_input_radio} \nAnswer_of_query:"""


              # Tokenize and generate answer
              inputs = tokenizer(input_prompt, return_tensors="pt")

              # Delaying the DaskDMatrix creation
              delayed_text_model_func = delayed(text_model)(inputs)  #da.from_array(z.toarray())

              # Computing the delayed object
              output = compute(delayed_text_model_func)

              #output = model.generate(**inputs, max_new_tokens=128)
              tokenizer_output = tokenizer.decode(output[0][0], skip_special_tokens=True)
              import re
              split_text = re.split(r"(Answer_of_query:)", tokenizer_output)

              # Output the sections

              answer_of_query = (split_text[2]).strip()

              results = answer_of_query
              st.write("Selected Entry:", entry_mapping[selected_entry])
              st.write("Your input for the entry:", user_input_radio)
              st.write(results)

    elif st.session_state.active_section == "Links List":
        st.header("Links List")
        links = [
            "https://www.example1.com",
            "https://www.example2.com",
            "https://www.example3.com"
        ]
        
        # Display links when button is pressed
        for link in links:
            st.markdown(f"[{link}]({link})")


if __name__ == "__main__":
    main()
