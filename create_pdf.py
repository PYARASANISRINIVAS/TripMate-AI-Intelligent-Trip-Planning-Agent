
import markdown
from weasyprint import HTML

def generate_pdf(markdown_content: str):

    html_content = markdown.markdown(
        markdown_content,
        extensions=["tables", "fenced_code"]
    )

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #1f3a5f;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #eee;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    HTML(string=html).write_pdf("trip_itinerary.pdf")

    return "trip_itinerary.pdf"
