
import markdown
from xhtml2pdf import pisa

def generate_pdf(markdown_content):

    html_content = markdown.markdown(
        markdown_content,
        extensions=["tables", "fenced_code"]
    )

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Helvetica;
                margin: 30px;
                line-height: 1.5;
            }}
            h1, h2, h3 {{
                color: #1f3a5f;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #999;
                padding: 8px;
            }}
            th {{
                background-color: #eeeeee;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    pdf_path = "trip_itinerary.pdf"

    with open(pdf_path, "wb") as file:
        result = pisa.CreatePDF(html, dest=file)

    if result.err:
        raise Exception("PDF generation failed")

    return pdf_path


# sample = """
# # TripPilot AI

# ## Delhi Trip

# - Duration: 3 Days
# - Travelers: 2

# ## Tourist Places

# | Place | Description |
# |---|---|
# | India Gate | Historic monument |
# | Red Fort | Historical fort |
# """


# generate_pdf(sample)