import gradio as gr

def video_identity(video):
    return video
# (type?, input, output)
webvideo = gr.Interface(video_identity, inputs=gr.Video(), outputs=gr.Video())

webvideo.launch(share=True)
