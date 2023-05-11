from flask import Flask, render_template, Response

class HttpStreamer():
    
    
    def __init__(self,camera ) -> None:
        self.app = Flask(__name__,
                        template_folder='../public/templates',
                        static_folder='../public/static',
                        )
    

        
        @self.app.route('/')
        def index():
            return render_template('index.html') 

        @self.app.route('/video_feed')
        def VideoStream():
            return Response(camera.get_recognize_face_stream(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
            # return Response(camera.get_frames_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

        

    def run(self):
        self.app.run('0.0.0.0')
    
        
        
    