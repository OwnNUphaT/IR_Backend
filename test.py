import requests

# === Flask backend URL ===
BASE_URL = "http://127.0.0.1:5000"

REGISTER_URL = f"{BASE_URL}/register"
LOGIN_URL = f"{BASE_URL}/login"

# === Test user credentials ===
TEST_USER = {
    "username": "ownny",
    "password": "own123"
}


# === Helper: Register test user ===
def register_user():
    print("\n[1] Registering test user...")
    response = requests.post(REGISTER_URL, json=TEST_USER)
    try:
        data = response.json()
        if data['status'] == 'success':
            print("✅ User registered successfully.")
        elif data['message'] == 'User already exists':
            print("ℹ️ User already exists. Proceeding to login test.")
        else:
            print(f"⚠️ Unexpected response: {data}")
    except Exception as e:
        print(f"❌ Error during registration: {e}")


# === Helper: Login with given credentials ===
def login_user(username, password, expected_status):
    print(f"\n[2] Attempting login with username='{username}' password='{password}'")
    response = requests.post(LOGIN_URL, json={"username": username, "password": password})
    try:
        data = response.json()
        status = data.get('status')
        message = data.get('message')

        if status == expected_status:
            print(f"✅ Test Passed. Status: {status}, Message: {message}")
        else:
            print(f"❌ Test Failed. Expected Status: {expected_status}, Got: {status}, Message: {message}")
    except Exception as e:
        print(f"❌ Error during login: {e}")


# === Run system tests ===
if __name__ == "__main__":
    # Start with registration
    register_user()

    # Test 1: Successful login
    login_user(TEST_USER['username'], TEST_USER['password'], expected_status="success")

    # Test 2: Invalid password
    login_user(TEST_USER['username'], "wrongpassword", expected_status="error")

    # Test 3: Missing fields
    print("\n[3] Attempting login with missing fields")
    response = requests.post(LOGIN_URL, json={"username": "", "password": ""})
    try:
        data = response.json()
        if data.get('status') == 'error':
            print(f"✅ Test Passed. Status: {data['status']}, Message: {data['message']}")
        else:
            print(f"❌ Test Failed. Unexpected response: {data}")
    except Exception as e:
        print(f"❌ Error during login: {e}")
