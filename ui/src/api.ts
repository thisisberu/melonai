const API_BASE_URL = 'http://127.0.0.1:8000'; 
const API_BASE_URL3 = " https://izg7bvt95ixl.share.zrok.io "
const API_BASE_URL2 = "https://dpds-health-wellness-api-223619550913.us-central1.run.app"

export async function sendChatMessage(text: string, userId: string, file?: File) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('user_id', userId);
  
  if (file) {
    formData.append('file', file);
  }

  const response = await fetch(`${API_BASE_URL}/chatbot/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
}

export async function sendRegisterMessage(text: string, userId: string, file?: File) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('user_id', userId);
  
  if (file) {
    formData.append('file', file);
  }

  const response = await fetch(`${API_BASE_URL}/register/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
}

export async function createNewChat(userId: string) {
  const formData = new FormData();
  formData.append('user_id', userId);

  const response = await fetch(`${API_BASE_URL}/newchat/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to create new chat');
  }

  return response.json();
}