from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import logging

# -------------------------------
# Configuration and Global Variables
# -------------------------------
SECRET_KEY = "your-very-long-secret-key-that-should-be-random"  # Replace with a secure key!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# For demonstration, we simulate key retrieval from Cloud KMS with an environment variable.
def get_encryption_key():
    key = os.getenv("ENCRYPTION_KEY")
    if key:
        return key.encode()  # Key should be 32 bytes for AES-256.
    else:
        # For demo purposes only – DO NOT use a fixed key in production!
        return b"0123456789abcdef0123456789abcdef"  # 32 bytes

# -------------------------------
# Audit Logging Setup (Immutable Ledger Simulation)
# -------------------------------
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
audit_handler = logging.FileHandler("audit.log", mode="a")  # Append-only
audit_formatter = logging.Formatter('%(asctime)s - %(message)s')
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)

def audit_log(message: str):
    audit_logger.info(message)

# -------------------------------
# OAuth2 / JWT Authentication Setup
# -------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simulated user database
fake_users_db = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderland",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
        "user_id": "user_1"
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    full_name: str | None = None
    disabled: bool | None = None
    user_id: str

class UserInDB(User):
    hashed_password: str

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def verify_password(plain_password, hashed_password):
    return fake_hash_password(plain_password) == hashed_password

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# -------------------------------
# AES-256 Encryption / Decryption Functions
# -------------------------------
def aes_encrypt(plaintext: bytes, key: bytes) -> bytes:
    # Use AES-256 in CBC mode with a random IV.
    iv = os.urandom(16)  # 16 bytes IV for AES block size
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # PKCS7 padding
    pad_len = 16 - (len(plaintext) % 16)
    padded_plaintext = plaintext + bytes([pad_len] * pad_len)
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return iv + ciphertext  # Prepend IV

def aes_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    iv = ciphertext[:16]
    actual_ciphertext = ciphertext[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
    pad_len = padded_plaintext[-1]
    return padded_plaintext[:-pad_len]

# -------------------------------
# Simulated Sensitive Data Store (e.g., Payment Details)
# -------------------------------
sensitive_data_store = {
    "user_1": {"name": "Alice Wonderland", "card_number": "4111111111111111"}
}

# -------------------------------
# FastAPI App Initialization
# -------------------------------
app = FastAPI()

# -------------------------------
# OAuth2 Token Endpoint
# -------------------------------
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    audit_log(f"User {user.username} logged in.")
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------------------
# Protected Endpoint to Retrieve Encrypted Sensitive Data
# -------------------------------
@app.get("/secure_data")
async def get_secure_data(current_user: User = Depends(get_current_active_user)):
    key = get_encryption_key()
    data = sensitive_data_store.get(current_user.user_id)
    if not data:
        raise HTTPException(status_code=404, detail="Sensitive data not found")
    plaintext = data["card_number"].encode()
    ciphertext = aes_encrypt(plaintext, key)
    audit_log(f"User {current_user.username} accessed secure data.")
    return {"encrypted_card_number": ciphertext.hex()}

# -------------------------------
# Endpoint to Decrypt Data (for testing; restrict in production)
# -------------------------------
@app.post("/decrypt_data")
async def decrypt_data(ciphertext_hex: str, current_user: User = Depends(get_current_active_user)):
    key = get_encryption_key()
    ciphertext = bytes.fromhex(ciphertext_hex)
    try:
        plaintext = aes_decrypt(ciphertext, key)
        return {"decrypted_data": plaintext.decode()}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Decryption failed")

# -------------------------------
# Right-to-be-Forgotten Endpoint (GDPR Compliance)
# -------------------------------
@app.delete("/right_to_be_forgotten/{user_id}")
async def right_to_be_forgotten(user_id: str, current_user: User = Depends(get_current_active_user)):
    if user_id in sensitive_data_store:
        del sensitive_data_store[user_id]
        audit_log(f"User {user_id} data deleted per GDPR request.")
        return {"status": "user data deleted"}
    else:
        raise HTTPException(status_code=404, detail="User data not found")

# -------------------------------
# Audit Log Retrieval Endpoint (for admin)
# -------------------------------
@app.get("/audit_log")
async def get_audit_log(current_user: User = Depends(get_current_active_user)):
    try:
        with open("audit.log", "r") as f:
            logs = f.read()
        return {"audit_log": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read audit log")

# -------------------------------
# Main: Run the App
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase-6:app", host="0.0.0.0", port=8001, reload=True)