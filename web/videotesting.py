import gradio as gr
# dont cry if the executable is not found. just download it bestie

def video_identity(video):
    return video
# (type?, input, output)
webvideo = gr.Interface(video_identity, inputs=gr.Video(), outputs=gr.Video())

webvideo.launch(share=True)
