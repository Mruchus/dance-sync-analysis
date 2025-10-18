from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import re
import shutil
import uuid

ALLOWED_EXTENSIONS = {'mp4', 'mov'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_score_from_output(output_text):
    """Extract synchronization score from dance.py output"""
    for line in output_text.split('\n'):
        if '% in sync' in line:
            match = re.search(r'(\d+\.\d+)%', line)
            if match:
                return float(match.group(1))
    return None


def reencode_video_for_browser(input_path, output_path):
    """Re-encode video for browser compatibility using FFmpeg"""
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart', '-y', output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg re-encoding failed: {e.stderr}")
        # fallback: copy original file
        shutil.copy(input_path, output_path)
        return False


# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # check if files were uploaded
    if 'reference' not in request.files or 'comparison' not in request.files:
        return jsonify({'error': 'Both videos are required'}), 400
    
    ref_file = request.files['reference']
    comp_file = request.files['comparison']
    
    if ref_file.filename == '' or comp_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not (allowed_file(ref_file.filename) and allowed_file(comp_file.filename)):
        return jsonify({'error': 'Invalid file type. Use MP4, MOV, or AVI'}), 400
    
    # create session directory
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    # save uploaded files
    ref_ext = os.path.splitext(ref_file.filename)[1]
    comp_ext = os.path.splitext(comp_file.filename)[1]
    
    ref_path = os.path.join(session_folder, f'reference{ref_ext}')
    comp_path = os.path.join(session_folder, f'comparison{comp_ext}')
    
    ref_file.save(ref_path)
    comp_file.save(comp_path)

    # create unique processing directory for this session
    session_output_dir = os.path.join(session_folder, 'processing')
    os.makedirs(session_output_dir, exist_ok=True)
    
    # run dance.py analysis
    try:
        cmd = ['python', 'dance.py', ref_path, comp_path, '--output-dir', session_output_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            return jsonify({
                'error': 'Processing failed',
                'details': result.stdout + '\n' + result.stderr,
                'logs': (result.stdout + '\n' + result.stderr).split('\n'),
            }), 500
        
        # extract score
        score = extract_score_from_output(result.stdout)

        # verify output video exists
        output_video = os.path.join(session_output_dir, 'output.mp4')
        if not os.path.exists(output_video):
            return jsonify({
                'error': 'Output video not found',
                'details': result.stdout,
            }), 500
        
        # re-encode output for browser compatibility
        final_video_path = os.path.join(session_folder, 'result.mp4')
        reencode_video_for_browser(output_video, final_video_path)
        
        return jsonify({
            'success': True,
            'score': score,
            'session_id': session_id,
            'output': result.stdout
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Processing timed out'}), 500
    except Exception as e:
        print(f"DEBUG: Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/video/<session_id>')
def video(session_id):
    """Stream video for playback in browser"""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    output_path = os.path.join(session_folder, 'result.mp4')
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(output_path, mimetype='video/mp4')


@app.route('/download/<session_id>')
def download(session_id):
    """Download video file"""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    output_path = os.path.join(session_folder, 'result.mp4')
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(output_path, as_attachment=True, download_name='dance_comparison.mp4')


@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup(session_id):
    """Delete session files"""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if os.path.exists(session_folder):
        shutil.rmtree(session_folder)
    return jsonify({'success': True})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8080)