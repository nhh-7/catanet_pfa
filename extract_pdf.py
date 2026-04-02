import PyPDF2

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += f"\n--- Page {page_num+1} ---\n"
            text += reader.pages[page_num].extract_text()
    return text

with open('catanet_text.txt', 'w', encoding='utf-8') as out:
    out.write(extract_text('CATANet.pdf'))
print("Extracted successfully.")
