from flask import Flask, render_template, Response
import os

class HttpStreamer():
    
    
    def __init__(self,detector,labels ) -> None:
        self.app = Flask(__name__,
                        template_folder=f'{os.getcwd()}/public/templates',
                        static_folder=f'{os.getcwd()}/public/static',
                        )


        
        @self.app.route('/')
        def index():
            return render_template('index.html') 

        @self.app.route('/video_feed')
        def video_feed():
            return Response(detector.start_regognition_all(labels),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
            # return Response(camera.get_frames_stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

        

    def run(self):
        self.app.run('0.0.0.0')
    
        
        
    