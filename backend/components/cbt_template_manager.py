from backend.database.mongo_connection import cbt_collection

class CBTTemplateManager:

    def get_template(self, template_name):
        template = cbt_collection.find_one({"name": template_name})
        return template

manager = CBTTemplateManager()
print(manager.get_template_by_emotion("high_anxiety"))
