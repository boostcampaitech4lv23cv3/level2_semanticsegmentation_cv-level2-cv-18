import requests, json

class NotionAutoWriter:
    def __init__(self, global_config:dict) -> None:
        self.api_info = global_config['notion_api']
        self.token  = self.api_info['token']
        self.database_id  = self.api_info['db_id']
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }

    def read_database(self) -> dict:
        '''return { 'status':status, 'data':data }'''
        readUrl = f"https://api.notion.com/v1/databases/{self.database_id}/query"
        res = requests.request("POST", readUrl, headers=self.headers)
        data = res.json()
        print(res.status_code)
        return{
            'status' : res.status_code,
            'data' : data
        }
    

    def post_page(self, title:str = 'untitled post', remark:str = '-', 
                        val_score:float = 0.0, test_score:float = 0.0,
                        wandb_link:str = '-', content:str = '-'
                ) -> int:
        createUrl = 'https://api.notion.com/v1/pages'
        databaseId = self.database_id
        headers = self.headers
        newPageData = {
            "parent": { "database_id": databaseId },
            "properties": {
                "Title": [
                            {
                                "type": "text",
                                "text": {
                                    "content": title,
                                },
                                "plain_text": title,
                            }
                        ],
                "Remark":[
                            {
                                "plain_text": remark,
                                "text": {
                                    "content": remark,
                                },
                                "type": "text"
                            }
                        ],
                "Status": {
                        "id": "123033cb-0438-4083-861a-01f3af3f27f8",
                            "name": "Not Submitted",
                            "color": "default"
                        },
                "Val Score": val_score,
                "Test Score": test_score,
                "WandB Link": wandb_link,
            },
            "children": [
                    {
                        "parent": { "database_id": databaseId },
                        "object": "block",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "text": {
                                        "content": content
                                    }
                                }
                            ]
                        }
                    },
                ]
        }

        data = json.dumps(newPageData)
        # print(str(uploadData))

        res = requests.request("POST", createUrl, headers=headers, data=data)

        if res.status_code != 200:
            print(res.status_code)
            print(res.text)
        return res.status_code
    
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
