// export interface Message {
//     id: string;
//     content: string;
//     sender: 'user' | 'bot';
//     timestamp: Date;
//     attachments?: string[];
//     files?: File[];
//   }
  
//   export interface ChatbotResponse {
//     response: string;
//   }
  
//   export interface ParsedResponse {
//     answer: string;
//     sources: string[];
//     moreTopic: string[];
//     products: Product[];
//   }
  
//   export interface Product {
//     name: string;
//     url: string;
//   }


export interface Message {
    id: string;
    content: string;
    sender: 'user' | 'bot';
    timestamp: Date;
    attachments?: string[];
    files?: File[];
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

export interface User {
  // ... other fields
  profile_picture?: string | File;
  // ... other fields
}