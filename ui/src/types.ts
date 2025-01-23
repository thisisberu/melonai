export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  attachments?: string[];
  files?: File[];  // Added files property to store the actual File objects
}

export interface Register{
  id: string; 
  first_name: string;
  last_name: string; 
  

}
export interface Chat {
  id: string;
  messages: Message[];
  createdAt: Date;
}

export interface ChatbotResponse {
  response: string;
}


export interface ParsedResponse {
  answer: string;
  sources: string[];
  moreTopic: string[];
  products: Product[];
}

export interface Product {
  name: string;
  url: string;
}