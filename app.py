import streamlit as st

# Example list of entries
entries = ["Entry 1", "Entry 2", "Entry 3", "Entry 4"]

# Create a dictionary to store the checkbox states
selected_entries = {}

st.header("Select Entries")
for entry in entries:
    selected_entries[entry] = st.checkbox(entry)

# Display selected entries
st.write("Selected Entries:")
selected_items = [entry for entry, selected in selected_entries.items() if selected]
st.write(selected_items)
