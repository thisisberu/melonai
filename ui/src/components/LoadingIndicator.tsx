// import React from 'react';
// import { Bot } from 'lucide-react';

// export function LoadingIndicator() {
//   return (
//     <div className="flex gap-4 max-w-[85%] animate-fadeIn">
//       <div className="relative">
//         <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg">
//           <Bot size={20} />
//         </div>
//         <div className="absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white bg-green-400" />
//       </div>
      
//       <div className="flex-1 bg-white dark:bg-gray-800 rounded-2xl px-6 py-4 shadow-md">
//         <div className="text-sm font-medium mb-2 flex items-center gap-2 text-gray-700 dark:text-gray-300">
//           AI Assistant
//           <span className="ml-2 flex gap-1">
//             <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
//             <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
//             <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" />
//           </span>
//         </div>
//         <div className="text-gray-600 dark:text-gray-400">Generating response...</div>
//       </div>
//     </div>
//   );
// }

import React from 'react';
import { Apple } from 'lucide-react';

export function LoadingIndicator() {
  return (
    <div className="flex gap-4 max-w-[85%] animate-fadeIn">
      <div className="relative">
        <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 
                      bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg">
          <Apple size={20} />
        </div>
        <div className="absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white bg-green-400" />
      </div>
      
      <div className="flex-1 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 
                    rounded-2xl px-6 py-4 shadow-md backdrop-blur-sm">
        <div className="text-sm font-medium mb-2 flex items-center gap-2 text-purple-700 dark:text-purple-300">
          NutriChat AI
          <span className="ml-2 flex gap-1">
            <span className="w-1.5 h-1.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full animate-bounce" />
          </span>
        </div>
        <div className="text-purple-600 dark:text-purple-300">Preparing your wellness insights...</div>
      </div>
    </div>
  );
}