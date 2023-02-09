import os
import shutil
from server import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
    path = os.path.join(os.getcwd(), 'tmp/')
    if os.path.exists(path):
        shutil.rmtree(path)
