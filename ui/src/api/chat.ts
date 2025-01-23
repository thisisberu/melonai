import axios from 'axios';
import type { ChatbotResponse } from '../types';

export async function sendMessage(text: string, userId: string, file?: File) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('user_id', userId);
  
  if (file) {
    formData.append('file', file);
  }

  const response = await axios.post<ChatbotResponse>('/api/chat', formData);
  return response.data;
}

export async function createNewChat(userId: string) {
  const response = await axios.post<{ chatId: string }>('/api/chat/new', { userId });
  return response.data;
}

