import requests, json
from utils import *

global_config = load_json('./config.json')

notion_api_info = global_config['notion_api']

token  = notion_api_info['token']
databaseId  = notion_api_info['db_id']

headers = {
    "Authorization": "Bearer " + notion_api_info['token'],
    "Content-Type": "application/json",
    "Notion-Version": "2021-05-13"
}

def readDatabase(databaseId, headers):
    readUrl = f"https://api.notion.com/v1/databases/{databaseId}/query"

    res = requests.request("POST", readUrl, headers=headers)
    data = res.json()
    print(res.status_code)
    # print(res.text)

    with open('./db.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)

def createPage(databaseId, headers):

    createUrl = 'https://api.notion.com/v1/pages'

    newPageData = {
        "parent": { "database_id": databaseId },
        "title" : "TEST",
        "properties": {
            "Id_TestName_No": [
                        {
                            "type": "text",
                            "text": {
                                "content": "API TEST",
                            },
                            
                            "plain_text": "API TEST",
                        }
                    ],
            "Status": {
                    "id": "123033cb-0438-4083-861a-01f3af3f27f8",
                        "name": "Not Submitted",
                        "color": "default"
                    },
            "Val miou": 0.1234,
            "Public miou": 0.1234,
        }
    }
    
    data = json.dumps(newPageData)
    # print(str(uploadData))

    res = requests.request("POST", createUrl, headers=headers, data=data)

    print(res.status_code)
    print(res.text)

def updatePage(pageId, headers):
    updateUrl = f"https://api.notion.com/v1/pages/{pageId}"

    updateData = {
        "properties": {
            "Value": {
                "rich_text": [
                    {
                        "text": {
                            "content": "Pretty Good"
                        }
                    }
                ]
            }        
        }
    }

    data = json.dumps(updateData)

    response = requests.request("PATCH", updateUrl, headers=headers, data=data)

    print(response.status_code)
    print(response.text)

# readDatabase(databaseId=databaseId, headers=headers)
print('done')

createPage(databaseId=databaseId, headers=headers)