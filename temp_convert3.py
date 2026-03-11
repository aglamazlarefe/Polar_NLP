import docx

def convert_to_csv():
    doc = docx.Document(r"d:\Polar_NLP\seyir defteri veri seti.docx")
    out = ["Metin,Etiket"]
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if '|' in text:
            parts = text.rsplit('|', 1)
            metin = parts[0].strip()
            etiket = parts[1].strip()
            if metin.lower() == 'metin' and etiket.lower() == 'etiket':
                continue
            metin = metin.replace('"', '""')
            out.append(f'"{metin}",{etiket}')
    
    with open(r"d:\Polar_NLP\output_utf8.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(out))

if __name__ == '__main__':
    convert_to_csv()
