import shutil
from IPython.display import HTML, display

def render(your_text):
    html_text = f'''
    <input type="hidden" value="{your_text}" id="clipboard-text">
    <button id="copy-button" onclick="copyToClipboard()" style="position: absolute; opacity: 0; cursor: pointer;">Copy text</button>
    <script>
    function copyToClipboard() {{
        var copyText = document.getElementById("clipboard-text");
        navigator.clipboard.writeText(copyText.value).then(function() {{
            console.log('Copying to clipboard was successful!');
        }}, function(err) {{
            console.error('Could not copy text: ', err);
        }});
    }}
    </script>
    '''
    display(HTML(html_text))

class TextRenderer:
    def __init__(self):
        self.texts = ["a", "b", "c"]

    def render_texts(self):
        for text in self.texts:
            render(text)