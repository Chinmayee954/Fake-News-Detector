from fastapi import  FastAPI 

app = FastAPI()

list_names = list()

@app.get("/{user_name}")
def write_home(user_name : str, query):
    return {
        "name" : user_name,
        "query" : query
    }

@app.put("/username/{user_name}")  
def put_data(user_name : str) :
    print(user_name)
    list_names.append(user_name)
    return {
        "name" : user_name
    }  

@app.post("/postdata")
def post_data(user_name : str) :
    list_names.append(user_name)
    return {
        "usernames" : list_names
    }  

@app.delete("/deletedata/{user_name}")
def delete_data(user_name : str) : 
    list_names.remove(user_name)
    return {
        "usernames" : list_names
    }  

@app.api_route("/homedata", methods=["GET", "POST", "DELETE", "PUT"])
def handledata(user_name : str):
    print(user_name)
    return {
        "username" : user_name
    }

   