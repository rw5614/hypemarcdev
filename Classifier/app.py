from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import urllib.parse
import pprint
import json
import tensorflow as tf
from keras.models import load_model, Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from pymongo import MongoClient
import urllib.parse
import pprint
from PIL import Image
import requests
import os
from io import BytesIO
import pandas as pd
from collections import Counter

class igUsers(object):
    userID = "-1"
    topCuisines = "-1"
    urlList = []
    toplabels = []
    topprobs = []

app = Flask(__name__)

def create_app():
    # Loads URL for Specific igUser
    def loadURL(igUser):
        if (len(igUser.urlList) == 0 ):
            print("User " + str(igUser.userID) + " has no URLs to download")
        else:
            print("Starting URL Download for user " + str(igUser.userID))
            #Clear Files
            for image in os.listdir("toClassify/"):
                print("Clearing " + os.path.join("toClassify/", image))
                os.remove(os.path.join("toClassify/", image))
            #Download URLs of Images for each user
            for i in range(0,len(igUser.urlList)):
                response = requests.get(igUser.urlList[i])
                img = Image.open(BytesIO(response.content))
                img.save("toClassify/"+repr(i)+".png")
                print("Added " + "toClassify/"+repr(i)+".png")
            print("Completed URL Download for user " + str(igUser.userID))

    # Performs Centered Crop on Image
    def crop_center(img,cropx,cropy):
        y, x, _ = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    # Loads Images Currently in Directory for classification
    def loadforclassify():
        resize_count = 0
        invalid_count = 0
        loaded_count = 0
        min_side = 299
        for image in os.listdir("toClassify"):
            try:
                #load image
                img_arr = plt.imread("toClassify/" + image)
                img_arr_rs = img_arr

                w, h, _ = img_arr.shape
                if (w, h != min_side):
                    if h > w :
                        wpercent = (min_side/float(w))
                        hsize = int((float(h)*float(wpercent)))
                        #print('new dims:', min_side, hsize)
                        img_arr_rs = skimage.transform.resize(img_arr, (min_side, hsize))
                        resize_count += 1
                        img_arr_rs = crop_center(img_arr_rs,299,299)
                    else:
                        hpercent = (min_side/float(h))
                        wsize = int((float(w)*float(hpercent)))
                        #print('new dims:', wsize, min_side)
                        img_arr_rs = skimage.transform.resize(img_arr, (wsize, min_side))
                        resize_count += 1
                        img_arr_rs = crop_center(img_arr_rs,299,299)
                all_imgs.append(img_arr_rs[:,:,:3])
                loaded_count += 1
            except:
                invalid_count+=1
        print("loaded:" + repr(loaded_count) + ",resized:" + repr(resize_count) + ",invalid:" + repr(invalid_count))

    # Preprocesses images 
    def preprocess_input(x):
        #x_copy = np.copy(x, dtype=np.float64)
        x_copy = np.array(x, dtype=np.float64)
        x_copy -= 0.5
        x_copy *= 2.0
        return x_copy
    # argmaxes for the top predictions and appends to user
    def find_top_pred(scores):
        for i in range(0,scores.shape[0]):
            top_label_ix = np.argmax(scores[i]) 
            # label 95 is Sushi
            confidence = scores[i][top_label_ix]
            print('Label: {}, Confidence: {}'.format(top_label_ix, confidence))
            toplabels.append(top_label_ix)
            topprobs.append(confidence)
    # convert numerical predictions to dish and assoc categories
    def getDish(dishlist):
        for i in range(0,len(dishlist)): 
            dishOut.append(data.dish[dishlist[i]])
            cuisineOut.append(data.cat1[toplabels[i]])
            cuisineOut.append(data.cat2[toplabels[i]])
        print("Appended Dishes")
    # updates cuisine field of matching igUser with userID matching database
    def updateUser(igUser):
        myquery = { "_id": igUser.userID }
        newvalues = { "$set": { "cuisines": igUser.topCuisines } }
        userCollection.update_one(myquery, newvalues)
        print("Updated Cuisines for" + str(igUser.userID))

    #Prepares TF Session and Loads Model
    sess = tf.Session()
    K.set_session(sess)
    model = load_model('model4b.10-0.68.hdf5')
    gd = sess.graph.as_graph_def()
    print(len(gd.node), 'Nodes')
    gd.node[:2]
    x = tf.placeholder(tf.float32, shape=model.get_input_shape_at(0))
    y = model(x)

    # Load Lookup Table
    data = pd.read_csv("foodlookup_v1.csv")

    # connect to database
    uri = "mongodb://tester:testing123@ds151614.mlab.com:51614/hypemarc-prod"
    client = MongoClient(uri)
    db = client['hypemarc-prod']
    userList = []

    # get collections for all Users and load into memory
    userCollection = db['userData']
    allUsers = userCollection.find({})

    print("[Grabbing Users from MongoDB]")
    for document in allUsers: # get each User in the collection of allUsers
        # TODO: add check that newUser not already existing in userList
        newUser = igUsers()
        newUser.userID = document.get('_id')
        print("Got User " + str(newUser.userID))
        userList.append(newUser)
    print("[Finished grabbing Users from MongoDB]")
    # grab all urls relevant to user from Food Database
    print("[Now Processing URLs for each User]")
    for user in userList:
        print("Processing " + str(user.userID))
        foodCollection = db['foodImages']
        allFood = foodCollection.find({'associatedUser':str(user.userID)})
        #clear URL List
        user.urlList = []
        for document in allFood:
            gotUrl = document.get('imageURL')
            if (gotUrl != None):
                user.urlList.append(gotUrl)
                print("Added:" + str(gotUrl))
            else:
                print("no relevant urls found to user " + str(user.userID))
    print("[Finished Processing URLs for each User]")
    print("[Now Processing Inference for each User]")
    for user in userList:
        print("Now inferring images for user " + str(user.userID))
        all_imgs = [] # unprocessed images
        proc_imgs = [] # processed images
        toplabels = [] # top category labels
        topprobs = [] # top probabilities
        dishOut = [] # text labels of dishes
        cuisineOut = [] # used for counting top n cuisines
        
        # download URLs
        loadURL(user)
        # classify images from this User
        loadforclassify()
        # preprocess images
        for i in range(0,len(all_imgs)):
            proc_imgs.append(preprocess_input(all_imgs[i]))
            #print("BLEH")
            #plt.imshow(proc_imgs[i])
        # some weird transformation thing for TensorFlow
        imgs = np.expand_dims(proc_imgs, 0)
        # imgs.shape #Verify Shape
        # Run inference session
        orig_scores = sess.run(y, feed_dict={x: imgs[0], K.learning_phase(): False})
        # process predicted outputs
        find_top_pred(orig_scores)
        # convert numerical predictions to dish text
        getDish(toplabels)
        # count results for dish text
        c = Counter(cuisineOut)
        # gets top cuisines for each user
        user.topCuisines = [item[0] for item in c.most_common(3)] #c.most_common(3)
        # call mongoDB to update user field
        updateUser(user)
    
@app.route('/')
def index():
    # connect to database
    uri = "mongodb://tester:testing123@ds151614.mlab.com:51614/hypemarc-prod"
    client = MongoClient(uri)
    db = client['hypemarc-prod']

    # get collections for all Users and load into memory
    userCollection = db['userData']
    allUsers = userCollection.find({ "username": "gcheung28" })
    resultsarr = ""
    profname = ""
    cuisines_detected = ""

    for document in allUsers: # get each User in the collection of allUsers
        resultsarr = resultsarr + str(document)
        profname = document.get('username')
        cuisines_detected = document.get('cuisines')
        biograph = document.get('bio')
        #print(json.dumps(str(document)))
    return render_template('index.html', profilename = profname, cuisines = cuisines_detected, biography = biograph, results = json.dumps(resultsarr))

@app.route('/recommended')
def recommendations():
    return render_template('index_static.html')

@app.route('/suggestions')
def suggestions():
    # # text = request.args.get('jsdata')
    suggestions_list = ["one", "two", "three"]

    # # if text:
    # #     r = requests.get('http://suggestqueries.google.com/complete/search?output=toolbar&hl=fr&q={}&gl=in'.format(text))

    # #     soup = BeautifulSoup(r.content, 'lxml')

    # #     suggestions = soup.find_all('suggestion')

    # #     for suggestion in suggestions:
    # #         suggestions_list.append(suggestion.attrs['data'])

    # #     print(suggestions_list)

    return render_template('suggestions.html', suggestions=suggestions_list)

if __name__ == '__main__':
    app.run(debug=True)