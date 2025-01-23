// import React, { useState, useEffect } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { ChatMessage } from '../ChatMessage';
// import { ChatInput } from '../ChatInput';
// import { LoadingIndicator } from '../LoadingIndicator';
// import { WellnessLogo } from '../WellnessLogo';
// import type { Message } from '../../types';

// type RegistrationStep = 
//   | 'welcome'
//   | 'name'
//   | 'email'
//   | 'phone'
//   | 'purpose'
//   | 'credentials';

// interface UserInfo {
//   name: string;
//   email: string;
//   phone: string;
//   purpose: string;
// }

// export function RegistrationChat() {
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [currentStep, setCurrentStep] = useState<RegistrationStep>('welcome');
//   const [isLoading, setIsLoading] = useState(false);
//   const [userInfo, setUserInfo] = useState<UserInfo>({
//     name: '',
//     email: '',
//     phone: '',
//     purpose: ''
//   });
//   const [showCredentialsForm, setShowCredentialsForm] = useState(false);
//   const navigate = useNavigate();

// //   const [regId, setRegId] = useState('1'); 

//   useEffect(() => {
//     // Initial welcome message
//     const welcomeMessage: Message = {
//       id: Date.now().toString(),
//       content: "👋 Hi! I'm here to help you create your account. Let's get started with a few questions. What's your name?",
//       sender: 'bot',
//       timestamp: new Date()
//     };
//     setMessages([welcomeMessage]);
//   }, []);

//   const handleResponse = async (text: string) => {
//     // Add user's message
//     const userMessage: Message = {
//       id: Date.now().toString(),
//       content: text,
//       sender: 'user',
//       timestamp: new Date()
//     };
//     setMessages(prev => [...prev, userMessage]);
//     setIsLoading(true);

//     // Process response based on current step
//     let botResponse = '';
//     let nextStep: RegistrationStep = currentStep;

//     switch (currentStep) {
//       case 'welcome':
//         setUserInfo(prev => ({ ...prev, name: text }));
//         botResponse = `Nice to meet you, ${text}! Could you please share your email address?`;
//         nextStep = 'email';
//         break;
//       case 'email':
//         if (!text.includes('@')) {
//           botResponse = "That doesn't look like a valid email address. Could you please try again?";
//         } else {
//           setUserInfo(prev => ({ ...prev, email: text }));
//           botResponse = "Great! Now, what's your phone number?";
//           nextStep = 'phone';
//         }
//         break;
//       case 'phone':
//         setUserInfo(prev => ({ ...prev, phone: text }));
//         botResponse = "Perfect! One last question: What's your primary purpose for joining our platform?";
//         nextStep = 'purpose';
//         break;
//       case 'purpose':
//         setUserInfo(prev => ({ ...prev, purpose: text }));
//         botResponse = `Excellent! Thanks for sharing that information. Now, let's set up your account credentials.
        
// Please click the "Create Account" button below to set your ABO ID and password.`;
//         nextStep = 'credentials';
//         // const newRegid = (parseInt(regId) + 1).toString(); 
//         // console.log("New regId = ",newRegid);
//         // setRegId(newRegid); 
//         // console.log("user info",userInfo) 
//         setShowCredentialsForm(true);
//         break;
//     }

//     // Add bot's response
//     const botMessage: Message = {
//       id: (Date.now() + 1).toString(),
//       content: botResponse,
//       sender: 'bot',
//       timestamp: new Date()
//     };
//     setTimeout(() => {
//       setMessages(prev => [...prev, botMessage]);
//       setCurrentStep(nextStep);
//       setIsLoading(false);
//     }, 1000);
//   };

//   return (
//     <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
//       {/* Header */}
//       <header className="bg-white dark:bg-gray-800 shadow-sm">
//         {/* <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
//           <WellnessLogo />
//         </div> */}
//       </header>

//       {/* Chat Messages */}
//       <div className="flex-1 overflow-y-auto px-4 py-8">
//         <div className="max-w-3xl mx-auto space-y-6">
//           {messages.map(message => (
//             <ChatMessage
//               key={message.id}
//               message={message}
//               onFollowUpClick={() => {}}
//             />
//           ))}
//           {isLoading && <LoadingIndicator />}
//         </div>
//       </div>

//       {/* Input Area */}
//       {!showCredentialsForm ? (
//         <div className="border-t border-gray-200 dark:border-gray-700">
//           <div className="max-w-3xl mx-auto">
//             <ChatInput onSend={handleResponse} disabled={isLoading} />
//           </div>
//         </div>
//       ) : (
//         <div className="border-t border-gray-200 dark:border-gray-700 p-4">
//           <div className="max-w-3xl mx-auto">
//             <button
//               onClick={() => navigate('/register',{ state: { userInfo } })}
//               className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
//             >
//               Create Account
//             </button>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChatMessage } from '../ChatMessage';
import { ChatInput } from '../ChatInput';
import { LoadingIndicator } from '../LoadingIndicator';
import { WellnessLogo } from '../WellnessLogo';
import { sendRegisterMessage } from '../../api';
import type { Message } from '../../types';
import { v4 as uuidv4 } from 'uuid'; 

type RegistrationStep = 
  | 'welcome'
  | 'name'
  | 'email'
  | 'phone'
  | 'purpose'
  | 'credentials';

// interface UserInfo {
//   name: string;
//   email: string;
//   phone: string;
//   purpose: string;
// }

interface UserInfo {
  // Personal Details
  full_name: string;
  date_of_birth: string;
  gender: string;
  ethnicity: string;
  profile_picture: File | null;
  
  // Communication Details
  mobile_number: string;
  email: string;
  address: string;
  
  // Identity Details
  national_id: string;
  identity_proof: File | null;
  
  // Sponsor Details
  sponsor_abo_id: string;
  sponsor_abo_name: string;
  
  // Payment Details
  card_number: string;
  card_name: string;
  
  // Final Verification
  id_proof_image: File | null;
}

export function RegistrationChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentStep, setCurrentStep] = useState<RegistrationStep>('welcome');
  const [isLoading, setIsLoading] = useState(false);
  // const [userInfo, setUserInfo] = useState<UserInfo>({
  //   name: '',
  //   email: '',
  //   phone: '',
  //   purpose: ''
  // });
  const [userInfo, setUserInfo] = useState<UserInfo>({
    // Personal Details
    full_name: '',
    date_of_birth: '',
    gender: '',
    ethnicity: '',
    profile_picture: null,
    
    // Communication Details
    mobile_number: '',
    email: '',
    address: '',
    
    // Identity Details
    national_id: '',
    identity_proof: null,
    
    // Sponsor Details
    sponsor_abo_id: '',
    sponsor_abo_name: '',
    
    // Payment Details
    card_number: '',
    card_name: '',
    
    // Final Verification
    id_proof_image: null
  });

  

  const [showCredentialsForm, setShowCredentialsForm] = useState(false);
  const navigate = useNavigate();
  const [chatId] = useState(() => uuidv4()); 
  useEffect(() => {
    // Initial welcome message
    const welcomeMessage: Message = {
      id: Date.now().toString(),
      content: "👋 Hi! I'm here to help you create your account. Let's get started with a few questions. What's your name?",
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const handleResponse = async (text: string, files?: FileList) => {
    // Add user's message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: text,
      sender: 'user',
      timestamp: new Date(),
      attachments: files ? Array.from(files).map(f => f.name) : undefined,
      files: files ? Array.from(files) : undefined
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send message to API
      const response = await sendRegisterMessage(
        text, 
        "2", // Using a fixed ID for registration flow
        files?.[0] // Send the first file if any
      );
      console.log("🚀 ~ handleResponse ~ response:", response)
      
      const isRegistrationComplete = response['response'].includes('Congratulations! Your registration is complete');
      if (isRegistrationComplete) {
        // console.log("🚀 ~ handleResponse ~ isRegistrationComplete:", isRegistrationComplete)
        setShowCredentialsForm(true); // Show the Create Account button

        if (response['user_info']) {
          console.log("🚀 ~ handleResponse ~ response['user_info']:", response['user_info'])
          console.log("hello");
           
          setUserInfo(response['user_info']);
          console.log("🚀 ~ handleResponse ~ user_info:", userInfo)
        }
          
      }



      // Add bot's response
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response['response'],
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Failed to get response:', error);
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I'm sorry, but I encountered an error. Could you please try again?",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateAccount = () => {
    navigate('/register', { 
      state: { 
        userInfo
      } 
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <WellnessLogo />
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map(message => (
            <ChatMessage
              key={message.id}
              message={message}
              onFollowUpClick={() => {}}
            />
          ))}
          {isLoading && <LoadingIndicator />}
        </div>
      </div>

      {/* Input Area */}
      {!showCredentialsForm ? (
        <div className="border-t border-gray-200 dark:border-gray-700">
          <div className="max-w-3xl mx-auto">
            <ChatInput onSend={handleResponse} disabled={isLoading} />
          </div>
        </div>
      ) : (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4">
          <div className="max-w-3xl mx-auto">
            <button
              onClick={handleCreateAccount}
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Create Account
            </button>
          </div>
        </div>
      )}
    </div>
  );
}