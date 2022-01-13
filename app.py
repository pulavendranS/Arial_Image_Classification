from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv

app = Flask(__name__)

dic = {0 : 'Field', 1 : 'Forest', 2 :"Grass", 3:'Industry', 4:'Parking', 5:'Resident', 6:'RiverLake'}


model = load_model('model/model.h5')
#model._make_predict_function()

def predict_label(img_path):
	#i = image.load_img(img_path, target_size=(256,256))
	img = cv.imread(img_path,0)
	img = cv.resize(img, (256,256))
	gabor_1 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F) 
	filtered_img_1 = cv.filter2D(img, cv.CV_8UC3, gabor_1)
	gabor_2 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F)
	filtered_img_2 = cv.filter2D(filtered_img_1, cv.CV_8UC3, gabor_2)
	img2 = cv.merge((filtered_img_2,filtered_img_2,filtered_img_2))
	#i = image.img_to_array(filtered_img_2)
	#i = i.reshape(1, 256,256,3)
	img = np.expand_dims(img2, axis=0)
	p = model.predict(img)
	x = np.argmax(p)
	return dic[x]
	#return p.any()


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"






@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("home.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)