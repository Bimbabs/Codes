import json
import os
from flask import Flask, request, render_template
import requests

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def place_order():
    try:
        if request.method == "POST":
            meal = request.form.get("meal")
            qty = request.form.get("qty")
            url = "https://lxkqocxlgd.execute-api.us-east-1.amazonaws.com/test_stage/order"
            headers = {"Content-Type": "application/json"}
            data_load = {"meal ": meal, "quantity ": qty}
            requests.request("POST", url, headers=headers, data=json.dumps(data_load, indent=4))
            return "Your order has been received"
        return render_template('/order.html')
    except Exception as e:
        return f"An Error Occured: {e}"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    # Action=SendMessage&MessageBody=$input.body


