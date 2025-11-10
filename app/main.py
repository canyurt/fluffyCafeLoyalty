import uvicorn
from . import app

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)

import os
print(os.path.exists("/secrets/firebase-service-account.json"))
