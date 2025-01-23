// import React, { useState, useRef, useEffect } from 'react';
// import { useAuth } from '../../hooks/useAuth';
// import { ChatInput } from '../ChatInput';
// import { ChatMessage } from '../ChatMessage';
// import { LoadingIndicator } from '../LoadingIndicator';
// import { ThemeToggle } from '../ThemeToggle';
// import { MessageCirclePlus, LogOut, User } from 'lucide-react';
// import { ProfileDropdown } from '../Profile/ProfileDropdown';
// import type { Message } from '../../types';
// import type { Chat } from '../../types'; 
// import { sendChatMessage, createNewChat } from '../../api'; 
// export function Chat() {

//   const [chats, setChats] = useState<Chat[]>([{
//     id: '1',
//     messages: [],
//     createdAt: new Date()
//   }]);
//   const [currentChatId, setCurrentChatId] = useState('1');
//   const [userId, setUserId] = useState('1');

//   const [messages, setMessages] = useState<Message[]>([]);
//   const [isLoading, setIsLoading] = useState(false);
//   const [isProfileOpen, setIsProfileOpen] = useState(false);
//   const profileButtonRef = useRef<HTMLButtonElement>(null);
//   const { user, logout } = useAuth();
//   const currentChat = chats.find(chat => chat.id === currentChatId)!; 
//   // Close profile dropdown when clicking outside
//   useEffect(() => {
//     function handleClickOutside(event: MouseEvent) {
//       if (
//         profileButtonRef.current &&
//         !profileButtonRef.current.contains(event.target as Node)
//       ) {
//         setIsProfileOpen(false);
//       }
//     }

//     document.addEventListener('mousedown', handleClickOutside);
//     return () => {
//       document.removeEventListener('mousedown', handleClickOutside);
//     };
//   }, []);

//   const handleSend = async (text: string, files?: FileList) => {
//     if (!text.trim() && (!files || files.length === 0)) return;
//     if (isLoading) return;

//     const userMessage: Message = {
//       id: Date.now().toString(),
//       content: text,
//       sender: 'user',
//       timestamp: new Date(),
//       attachments: files ? Array.from(files).map(f => f.name) : undefined,
//       files: files ? Array.from(files) : undefined
//     };

//     setChats(prevChats => prevChats.map(chat => {
//       if (chat.id === currentChatId) {
//         return {
//           ...chat,
//           messages: [...chat.messages, userMessage]
//         };
//       }
//       return chat;
//     }));
//     setIsLoading(true);

//     try {
//       if (!user?.aboId) return;
//       const response = await sendChatMessage(
//         text,
//         user.aboId,
//         files && files.length > 0 ? files[0] : undefined
//       );

//       const botMessage: Message = {
//         id: (Date.now() + 1).toString(),
//         content: response.response,
//         sender: 'bot',
//         timestamp: new Date()
//       };

//       setChats(prevChats => prevChats.map(chat => {
//         if (chat.id === currentChatId) {
//           return {
//             ...chat,
//             messages: [...chat.messages, botMessage]
//           };
//         }
//         return chat;
//       }));
//     } catch (error) {
//       console.error('Failed to get response:', error);
//     } finally {
//       setIsLoading(false);
//     }
//   };


//   const handleNewChat = async () => {
//     if (isLoading) return;
    
//     try {
//       setIsLoading(true);
//       const newUserId = (parseInt(userId) + 1).toString();
//       setUserId(newUserId);
//       await createNewChat(newUserId);

//       const newChat: Chat = {
//         id: Date.now().toString(),
//         messages: [],
//         createdAt: new Date()
//       };
      
//       setChats(prevChats => [...prevChats, newChat]);
//       setCurrentChatId(newChat.id);
//     } catch (error) {
//       console.error('Failed to create new chat:', error);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleFollowUpClick = async (question: string) => {
//     if (isLoading) return;
    
//     setTimeout(() => {
//       window.scrollTo({
//         top: document.documentElement.scrollHeight,
//         behavior: 'smooth'
//       });
//     }, 100);

//     await handleSend(question);
//   };

//   return (
//     <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900">
//       {/* Header */}
//       <header className="bg-white dark:bg-gray-800 shadow-sm">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
//           <h1 className="text-xl font-semibold text-gray-900 dark:text-white">EatsAI🍕🥗</h1>
//           <div className="flex items-center space-x-4">
//             <button
//               onClick={handleNewChat}
//               className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
//               title="New chat"
//             >
//               <MessageCirclePlus size={20} />
//             </button>
//             <ThemeToggle />
//             <div className="relative">
//               <button
//                 ref={profileButtonRef}
//                 onClick={() => setIsProfileOpen(!isProfileOpen)}
//                 className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
//                 title="View profile"
//               >
//                 <User size={20} />
//               </button>
//               <ProfileDropdown 
//                 isOpen={isProfileOpen} 
//                 onClose={() => setIsProfileOpen(false)} 
//               />
//             </div>
//             <button
//               onClick={logout}
//               className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
//               title="Sign out"
//             >
//               <LogOut size={20} />
//             </button>
//           </div>
//         </div>
//       </header>

//       {/* Chat Messages */}
//       <div className="flex-1 overflow-y-auto px-4 py-8">
//         <div className="max-w-3xl mx-auto space-y-6">
//           {currentChat.messages.map(message => (
//             <ChatMessage
//               key={message.id}
//               message={message}
//               onFollowUpClick={handleFollowUpClick}
//             />
//           ))}
//           {isLoading && <LoadingIndicator />}
//           {currentChat.messages.length === 0 && (
//             <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
//               <p className="text-lg">Welcome, {user?.aboId}! 👋</p>
//               <p className="mt-2">Start a conversation by typing a message below.</p>
//             </div>
//           )}
//         </div>
//       </div>

//       {/* Chat Input */}
//       <div className="border-t border-gray-200 dark:border-gray-700">
//         <div className="max-w-3xl mx-auto">
//           <ChatInput onSend={handleSend} disabled={isLoading} />
//         </div>
//       </div>

//     </div>
//   );
// }


import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { ChatInput } from '../ChatInput';
import { ChatMessage } from '../ChatMessage';
import { LoadingIndicator } from '../LoadingIndicator';
import { ThemeToggle } from '../ThemeToggle';
import { MessageCirclePlus, LogOut, User } from 'lucide-react';
import { ProfileDropdown } from '../Profile/ProfileDropdown';
import type { Message } from '../../types';

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const profileButtonRef = useRef<HTMLButtonElement>(null);
  const { user, logout } = useAuth();

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        profileButtonRef.current &&
        !profileButtonRef.current.contains(event.target as Node)
      ) {
        setIsProfileOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSend = async (text: string, files?: FileList) => {
    if (!text.trim() && (!files || files.length === 0)) return;

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
      const response = await new Promise<string>(resolve => 
        setTimeout(() => resolve("This is a sample response from the AI.\n\nSources:\n- https://example.com\n\nWould you like to know more about:\n• Topic 1\n• Topic 2"), 1000)
      );

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Failed to get response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleFollowUpClick = (question: string) => {
    handleSend(question);
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-orange-50 dark:from-purple-950 dark:via-pink-950 dark:to-orange-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <h1 className="text-xl font-semibold bg-gradient-to-r from-purple-600 to-pink-600 dark:from-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
            NutriChat AI
          </h1>
          <div className="flex items-center space-x-4">
            <button
              onClick={handleNewChat}
              className="p-2 text-purple-500 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300 
                       hover:bg-purple-100 dark:hover:bg-purple-800/50 rounded-lg transition-all duration-200"
              title="New chat"
            >
              <MessageCirclePlus size={20} />
            </button>
            <ThemeToggle />
            <div className="relative">
              <button
                ref={profileButtonRef}
                onClick={() => setIsProfileOpen(!isProfileOpen)}
                className="p-2 text-pink-500 hover:text-pink-700 dark:text-pink-400 dark:hover:text-pink-300
                         hover:bg-pink-100 dark:hover:bg-pink-800/50 rounded-lg transition-all duration-200"
                title="View profile"
              >
                <User size={20} />
              </button>
              <ProfileDropdown 
                isOpen={isProfileOpen} 
                onClose={() => setIsProfileOpen(false)} 
              />
            </div>
            <button
              onClick={logout}
              className="p-2 text-orange-500 hover:text-orange-700 dark:text-orange-400 dark:hover:text-orange-300
                       hover:bg-orange-100 dark:hover:bg-orange-800/50 rounded-lg transition-all duration-200"
              title="Sign out"
            >
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map(message => (
            <ChatMessage
              key={message.id}
              message={message}
              onFollowUpClick={handleFollowUpClick}
            />
          ))}
          {isLoading && <LoadingIndicator />}
          {messages.length === 0 && (
            <div className="text-center mt-8">
              <div className="inline-block p-4 rounded-2xl bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm shadow-xl">
                <p className="text-lg font-medium bg-gradient-to-r from-purple-600 to-pink-600 dark:from-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
                  Welcome, {user?.aboId}! 👋
                </p>
                <p className="mt-2 text-gray-600 dark:text-gray-300">
                  Start your wellness journey by asking a question below.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chat Input */}
      <div className="border-t border-purple-200/50 dark:border-purple-700/50">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={handleSend} disabled={isLoading} />
        </div>
      </div>
    </div>
  );
}