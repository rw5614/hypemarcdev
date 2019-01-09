from pymongo import MongoClient
import urllib.parse
import pprint

# connect to database
uri = "mongodb://tester:testing123@ds151614.mlab.com:51614/hypemarc-prod"
client = MongoClient(uri)
db = client['hypemarc-prod']

# get collections
userCollection = db['userData']
allUsers = userCollection.find({})
for document in allUsers: # get each entry in the collection
	print(document)

foodCollection = db['foodImages']
allFood = foodCollection.find({})
for document in allFood:
	print(document)
