// import React, { useState } from 'react';
// import { Send } from 'lucide-react';
// import { FileUpload } from './FileUpload';

// interface ChatInputProps {
//   onSend: (message: string, files?: FileList) => void;
//   disabled?: boolean;
// }

// export function ChatInput({ onSend, disabled }: ChatInputProps) {
//   const [message, setMessage] = useState('');
//   const [files, setFiles] = useState<FileList | null>(null);

//   const handleSubmit = (e: React.FormEvent) => {
//     e.preventDefault();
//     if ((message.trim() || files) && !disabled) {
//       onSend(message, files || undefined);
//       setMessage('');
//       setFiles(null);
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="border-t border-gray-200 p-4 bg-white shadow-lg">
//       <div className="flex items-center gap-3 max-w-3xl mx-auto">
//         <FileUpload onFileSelect={setFiles} disabled={disabled} />
//         <input
//           type="text"
//           value={message}
//           onChange={(e) => setMessage(e.target.value)}
//           placeholder="Type your message..."
//           className="flex-1 p-3 border border-gray-200 rounded-xl focus:outline-none focus:border-black focus:ring-1 focus:ring-black disabled:opacity-50 transition-all"
//           disabled={disabled}
//         />
//         <button
//           type="submit"
//           disabled={disabled || (!message.trim() && !files)}
//           className="p-3 bg-black text-white rounded-xl hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
//         >
//           <Send size={20} />
//         </button>
//       </div>
//       {files && (
//         <div className="mt-2 max-w-3xl mx-auto">
//           <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded-lg">
//             {Array.from(files).map((file, index) => (
//               <div key={index} className="inline-block mr-3">
//                 • {file.name}
//               </div>
//             ))}
//           </div>
//         </div>
//       )}
//     </form>
//   );
// }


import React, { useState } from 'react';
import { Send, Upload } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string, files?: FileList) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<FileList | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((message.trim() || files) && !disabled) {
      onSend(message, files || undefined);
      setMessage('');
      setFiles(null);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 shadow-lg rounded-t-xl">
      <div className="flex items-center gap-3 max-w-3xl mx-auto">
        <label className="cursor-pointer">
          <input
            type="file"
            className="hidden"
            onChange={(e) => setFiles(e.target.files)}
            multiple
            disabled={disabled}
          />
          <div className="p-3 hover:bg-green-100 dark:hover:bg-green-800/50 rounded-full transition-colors">
            <Upload size={20} className="text-green-600 dark:text-green-400" />
          </div>
        </label>
        
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about nutrition and wellness..."
          className="flex-1 p-3 border border-green-200 dark:border-green-800 rounded-xl 
                   focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 
                   disabled:opacity-50 transition-all bg-white dark:bg-gray-800
                   placeholder-green-400 dark:placeholder-green-500"
          disabled={disabled}
        />
        
        <button
          type="submit"
          disabled={disabled || (!message.trim() && !files)}
          className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-xl 
                   hover:from-green-600 hover:to-emerald-600 transition-all duration-300 
                   disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          <Send size={20} />
        </button>
      </div>
      
      {files && (
        <div className="mt-2 max-w-3xl mx-auto">
          <div className="text-sm text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30 p-2 rounded-lg">
            {Array.from(files).map((file, index) => (
              <div key={index} className="inline-block mr-3">
                • {file.name}
              </div>
            ))}
          </div>
        </div>
      )}
    </form>
  );
}