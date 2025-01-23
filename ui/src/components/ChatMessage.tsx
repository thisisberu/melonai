// import React, { useState } from 'react';
// import { FileText, Bot, User, ExternalLink, MessageCircle, Clock, ChevronDown, ShoppingBag } from 'lucide-react';
// import { useTypewriter } from '../hooks/useTypewriter';
// import ReactMarkdown from 'react-markdown';
// import remarkGfm from 'remark-gfm';
// import rehypeRaw from 'rehype-raw';
// import rehypeHighlight from 'rehype-highlight';
// import type { Message, ParsedResponse, Product } from '../types';
 
// interface ChatMessageProps {
//   message: Message;
//   onFollowUpClick?: (question: string) => void;
// }
 
// function parseContent(response: string): ParsedResponse {
  
//   // console.log("🚀 ~ parseContent ~ response:",(response));
//   try {
    
//     // First, attempt to parse the input as a JavaScript object
//     const parsedObject = eval('(' + response + ')');
   
//     // If parsing succeeds, extract information directly from the object
//     return {
//       answer: parsedObject.answer || "",
//       sources: Array.isArray(parsedObject.sources) ? parsedObject.sources : [],
//       moreTopic: Array.isArray(parsedObject.more_topics) ? parsedObject.more_topics : [],
//       products: parsedObject.products ? parseProducts(parsedObject.products) : []
//     };
//   } catch (objectParseError) {
//     // Fallback to the original line-by-line parsing method
//     const result: ParsedResponse = {
//       answer: "",
//       sources: [],
//       moreTopic: [],
//       products: []
//     };
   
//     const lines = response.split("\n").map((line) => line.trim()).filter(Boolean);
//     let currentSection: keyof ParsedResponse | 'none' = "answer";
   
//     for (const line of lines) {
//       if (line.toLowerCase().startsWith("sources:")) {
//         currentSection = "sources";
//         continue;
//       } else if (line.toLowerCase().startsWith("would you like to know more about:")) {
//         currentSection = "moreTopic";
//         continue;
//       } else if (line.toLowerCase().includes("recommended products:")) {
//         currentSection = "products";
//         continue;
//       } else if (line.startsWith("---")) {
//         currentSection = "none";
//         continue;
//       }
   
//       switch (currentSection) {
//         case "answer":
//           result.answer = result.answer
//             ? result.answer + "\n" + line
//             : line;
//           break;
//         case "sources":
//           if (line.startsWith("-") || line.startsWith("•")) {
//             result.sources.push(line.slice(1).trim());
//           } else if (!line.toLowerCase().startsWith("sources:")) {
//             result.sources.push(line.trim());
//           }
//           break;
//         case "moreTopic":
//           if (line.startsWith("-") || line.startsWith("•")) {
//             result.moreTopic.push(line.slice(1).trim());
//           } else if (!line.toLowerCase().startsWith("would you like to know more about")) {
//             result.moreTopic.push(line.trim());
//           }
//           break;
//         case "products":
//           if (line.startsWith("-") || line.startsWith("•")) {
//             const match = line.match(/\[(.*?)\]\((.*?)\)/);
//             if (match) {
//               result.products.push({
//                 name: match[1],
//                 url: match[2]
//               });
//             }
//           }
//           break;
//       }
//     }
//     return result;
//   }
// }
 
// // Helper function to parse products from markdown string
// function parseProducts(productsString: string): Product[] {
//   const productsArray: Product[] = [];
//   const productLines = productsString.split('\n');
 
//   productLines.forEach(line => {
//     const match = line.match(/\[(.*?)\]\((.*?)\)/);
//     if (match) {
//       productsArray.push({
//         name: match[1],
//         url: match[2]
//       });
//     }
//   });
 
//   return productsArray;
// }
 
 
// const isImageFile = (filename: string): boolean => {
//   const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'];
//   return imageExtensions.some(ext => filename.toLowerCase().endsWith(ext));
// };
 
// export function ChatMessage({ message, onFollowUpClick }: ChatMessageProps) {
//   const [isSourcesExpanded, setIsSourcesExpanded] = useState(false);
//   const [isProductsExpanded, setIsProductsExpanded] = useState(false);
//   const isBot = message.sender === 'bot';
//   // console.log("message here: ", message)
//   const { answer, sources, moreTopic, products } = parseContent(message.content);
//   const { displayedText, isTyping } = useTypewriter(answer, 30);
 
//   const handleFollowUpClick = (question: string) => {
//     if (onFollowUpClick) {
//       onFollowUpClick(question);
//     }
//   };
 
//   const renderContent = () => {
//     const textToRender = isBot ? displayedText : message.content;
//     if (!textToRender) return null;
 
//     return (
//       <ReactMarkdown
//         className={`prose prose-sm dark:prose-invert max-w-none ${
//           !isBot && 'text-white/100 prose-headings:text-white/100 prose-strong:text-white/100 prose-em:text-white/100'
//         }`}
//         remarkPlugins={[remarkGfm]}
//         rehypePlugins={[rehypeRaw, rehypeHighlight]}
//       >
//         {textToRender}
//       </ReactMarkdown>
//     );
//   };
 
//   return (
//     <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-8 group animate-fadeIn`}>
//       <div className={`flex gap-4 max-w-[85%] relative`}>
//         <div className="relative">
//           <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0
//             ${isBot ? 'bg-gradient-to-br from-blue-500 to-purple-600' : 'bg-gradient-to-br from-gray-700 to-gray-900'}
//             text-white shadow-lg`}
//           >
//             {isBot ? <Bot size={20} /> : <User size={20} />}
//           </div>
//           <div className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white
//             ${isBot ? 'bg-green-400' : 'bg-blue-400'}`} />
//         </div>
 
//         <div className={`flex-1 ${
//           isBot
//             ? 'bg-white dark:bg-gray-800'
//             : 'bg-gradient-to-br from-blue-500 to-blue-600'
//           } rounded-2xl px-6 py-4 shadow-md hover:shadow-lg transition-shadow`}>
         
//           <div className={`text-sm font-medium mb-2 flex items-center gap-2
//             ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}`}>
//             {isBot ? (
//               <>
//                 AI Assistant
//                 {isTyping && (
//                   <span className="ml-2 flex gap-1">
//                     <span className="w-1 h-1 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
//                     <span className="w-1 h-1 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
//                     <span className="w-1 h-1 bg-blue-500 rounded-full animate-bounce" />
//                   </span>
//                 )}
//               </>
//             ) : 'You'}
//             <span className="text-xs opacity-50">•</span>
//             <span className="text-xs flex items-center gap-1 opacity-50">
//               <Clock size={12} />
//               {new Date(message.timestamp).toLocaleTimeString([], {
//                 hour: '2-digit',
//                 minute: '2-digit'
//               })}
//             </span>
//           </div>
 
//           <div className={`space-y-4 text-base leading-relaxed ${
//             isBot ? 'text-gray-800 dark:text-gray-200' : 'text-white/100'
//           }`}>
//             {renderContent()}
//           </div>
 
//           {/* Sources Section */}
//           {!isTyping && sources.length > 0 && (
//             <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
//               <button
//                 onClick={() => setIsSourcesExpanded(!isSourcesExpanded)}
//                 className={`w-full flex items-center justify-between text-sm font-medium mb-2
//                   ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}
//                   hover:opacity-80 transition-opacity`}
//               >
//                 <span className="flex items-center gap-2">
//                   <ExternalLink size={16} />
//                   Sources ({sources.length})
//                 </span>
//                 <ChevronDown
//                   size={16}
//                   className={`transform transition-transform duration-200 ${
//                     isSourcesExpanded ? 'rotate-180' : ''
//                   }`}
//                 />
//               </button>
//               {isSourcesExpanded && (
//                 <div className="space-y-2 mt-2 animate-fadeIn">
//                   {sources.map((source, index) => (
//                     <a
//                       key={index}
//                       href={source}
//                       target="_blank"
//                       rel="noopener noreferrer"
//                       className={`flex items-center gap-2 text-sm group/link
//                         ${isBot
//                           ? 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300'
//                           : 'text-white hover:text-white/90'}`}
//                     >
//                       <ExternalLink size={16} className="group-hover/link:translate-x-0.5 transition-transform" />
//                       <span className="underline underline-offset-2">{source}</span>
//                     </a>
//                   ))}
//                 </div>
//               )}
//             </div>
//           )}
 
//           {/* Products Section */}
//           {!isTyping && products.length > 0 && (
//             <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
//               <button
//                 onClick={() => setIsProductsExpanded(!isProductsExpanded)}
//                 className={`w-full flex items-center justify-between text-sm font-medium mb-2
//                   ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}
//                   hover:opacity-80 transition-opacity`}
//               >
//                 <span className="flex items-center gap-2">
//                   <ShoppingBag size={16} />
//                   Recommended Products ({products.length})
//                 </span>
//                 <ChevronDown
//                   size={16}
//                   className={`transform transition-transform duration-200 ${
//                     isProductsExpanded ? 'rotate-180' : ''
//                   }`}
//                 />
//               </button>
//               {isProductsExpanded && (
//                 <div className="space-y-2 mt-2 animate-fadeIn">
//                   {products.map((product, index) => (
//                     <a
//                       key={index}
//                       href={product.url}
//                       target="_blank"
//                       rel="noopener noreferrer"
//                       className={`flex items-center gap-2 text-sm group/link
//                         ${isBot
//                           ? 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300'
//                           : 'text-white hover:text-white/90'}`}
//                     >
//                       <ShoppingBag size={16} className="group-hover/link:translate-x-0.5 transition-transform" />
//                       <span className="underline underline-offset-2">{product.name}</span>
//                     </a>
//                   ))}
//                 </div>
//               )}
//             </div>
//           )}
 
//           {/* Follow-up Questions */}
//           {!isTyping && isBot && moreTopic.length > 0 && (
//             <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
//               <div className="text-sm font-medium mb-3 text-gray-700 dark:text-gray-300">
//                 Would you like to know more about:
//               </div>
//               <div className="flex flex-wrap gap-2">
//                 {moreTopic.map((question, index) => (
//                   <button
//                     key={index}
//                     onClick={() => handleFollowUpClick(question)}
//                     className="flex items-center gap-2 text-sm bg-gray-100 dark:bg-gray-700/50
//                              hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full px-4 py-2.5
//                              transition-all duration-200 cursor-pointer text-gray-700 dark:text-gray-300
//                              hover:shadow-md active:scale-95"
//                   >
//                     <MessageCircle size={16} className="text-blue-500" />
//                     <span>{question}</span>
//                   </button>
//                 ))}
//               </div>
//             </div>
//           )}
 
//           {/* Attachments */}
//           {message.attachments && message.attachments.length > 0 && (
//             <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
//               <div className={`text-sm font-medium mb-2
//                 ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}`}>
//                 Attachments
//               </div>
//               <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
//                 {message.attachments.map((file, index) => (
//                   <div
//                     key={index}
//                     className={`flex items-start gap-3 p-3 rounded-lg group/attachment
//                       ${isBot
//                         ? 'bg-gray-50 dark:bg-gray-700/50 text-gray-600 dark:text-gray-300'
//                         : 'bg-white/10 text-white'}`}
//                   >
//                     {isImageFile(file) ? (
//                       <div className="relative group/image">
//                         <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-600 flex items-center justify-center">
//                           <img
//                             src={URL.createObjectURL(message.files?.[index] as Blob)}
//                             alt={file}
//                             className="w-full h-full object-cover"
//                             onError={(e) => {
//                               const target = e.target as HTMLImageElement;
//                               target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxyZWN0IHg9IjMiIHk9IjMiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgcng9IjIiIHJ5PSIyIi8+PGNpcmNsZSBjeD0iOC41IiBjeT0iOC41IiByPSIxLjUiLz48cG9seWxpbmUgcG9pbnRzPSIyMSAxNSAxNiAxMCA1IDIxIi8+PC9zdmc+';
//                             }}
//                           />
//                         </div>
//                         <div className="hidden group-hover/image:block absolute left-20 bottom-0 z-10">
//                           <img
//                             src={URL.createObjectURL(message.files?.[index] as Blob)}
//                             alt={file}
//                             className="max-w-sm rounded-lg shadow-xl border-2 border-white dark:border-gray-700"
//                             onError={(e) => {
//                               const target = e.target as HTMLImageElement;
//                               target.style.display = 'none';
//                             }}
//                           />
//                         </div>
//                       </div>
//                     ) : (
//                       <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-gray-600 flex items-center justify-center flex-shrink-0">
//                         <FileText size={20} className="text-gray-500 dark:text-gray-400" />
//                       </div>
//                     )}
//                     <div className="flex-1 min-w-0">
//                       <p className="text-sm font-medium truncate">{file}</p>
//                       {message.files?.[index] && (
//                         <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
//                           {(message.files[index] as File).size > 1024 * 1024
//                             ? `${((message.files[index] as File).size / (1024 * 1024)).toFixed(1)} MB`
//                             : `${((message.files[index] as File).size / 1024).toFixed(1)} KB`}
//                         </p>
//                       )}
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }


import React, { useState } from 'react';
import { FileText, Bot, User, ExternalLink, MessageCircle, Clock, ChevronDown, ShoppingBag, Apple } from 'lucide-react';
import { useTypewriter } from '../hooks/useTypewriter';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeHighlight from 'rehype-highlight';
import type { Message, ParsedResponse, Product } from '../types';

interface ChatMessageProps {
  message: Message;
  onFollowUpClick?: (question: string) => void;
}

function parseContent(response: string): ParsedResponse {
  const result: ParsedResponse = { 
    answer: "", 
    sources: [], 
    moreTopic: [],
    products: []
  };
  
  const lines = response.split("\n").map((line) => line.trim()).filter(Boolean);
  let currentSection: keyof ParsedResponse | 'none' = "answer";
  
  for (const line of lines) {
    if (line.toLowerCase().startsWith("sources:")) {
      currentSection = "sources";
      continue;
    } else if (line.toLowerCase().startsWith("would you like to know more about:")) {
      currentSection = "moreTopic";
      continue;
    } else if (line.toLowerCase().includes("recommended products:")) {
      currentSection = "products";
      continue;
    } else if (line.startsWith("---")) {
      currentSection = "none";
      continue;
    }

    switch (currentSection) {
      case "answer":
        result.answer = result.answer 
          ? result.answer + "\n" + line 
          : line;
        break;
      case "sources":
        if (line.startsWith("-") || line.startsWith("•")) {
          result.sources.push(line.slice(1).trim());
        } else if (!line.toLowerCase().startsWith("sources:")) {
          result.sources.push(line.trim());
        }
        break;
      case "moreTopic":
        if (line.startsWith("-") || line.startsWith("•")) {
          result.moreTopic.push(line.slice(1).trim());
        } else if (!line.toLowerCase().startsWith("would you like to know more about")) {
          result.moreTopic.push(line.trim());
        }
        break;
      case "products":
        if (line.startsWith("-") || line.startsWith("•")) {
          const match = line.match(/\[(.*?)\]\((.*?)\)/);
          if (match) {
            result.products.push({
              name: match[1],
              url: match[2]
            });
          }
        }
        break;
    }
  }
  return result;
}

const isImageFile = (filename: string): boolean => {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'];
  return imageExtensions.some(ext => filename.toLowerCase().endsWith(ext));
};

export function ChatMessage({ message, onFollowUpClick }: ChatMessageProps) {
  const [isSourcesExpanded, setIsSourcesExpanded] = useState(false);
  const [isProductsExpanded, setIsProductsExpanded] = useState(false);
  const isBot = message.sender === 'bot';
  const { answer, sources, moreTopic, products } = parseContent(message.content);
  const { displayedText, isTyping } = useTypewriter(answer, 30);

  const renderContent = () => {
    const textToRender = isBot ? displayedText : message.content;
    if (!textToRender) return null;

    return (
      <ReactMarkdown
        className={`prose prose-sm dark:prose-invert max-w-none ${
          !isBot && 'text-white/100 prose-headings:text-white/100 prose-strong:text-white/100 prose-em:text-white/100'
        }`}
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeHighlight]}
      >
        {textToRender}
      </ReactMarkdown>
    );
  };

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-8 group animate-fadeIn`}>
      <div className={`flex gap-4 max-w-[85%] relative`}>
        <div className="relative">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 
            ${isBot ? 'bg-gradient-to-br from-green-400 to-emerald-500' : 'bg-gradient-to-br from-orange-400 to-orange-500'} 
            text-white shadow-lg transform transition-transform hover:scale-105`}
          >
            {isBot ? <Apple size={20} /> : <User size={20} />}
          </div>
          <div className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white
            ${isBot ? 'bg-green-400' : 'bg-orange-400'}`} />
        </div>

        <div className={`flex-1 ${
          isBot 
            ? 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20' 
            : 'bg-gradient-to-br from-orange-500 to-yellow-500'
          } rounded-2xl px-6 py-4 shadow-md hover:shadow-lg transition-all duration-300`}>
          
          <div className={`text-sm font-medium mb-2 flex items-center gap-2
            ${isBot ? 'text-green-700 dark:text-green-300' : 'text-white'}`}>
            {isBot ? (
              <>
                Nutrition Assistant
                {isTyping && (
                  <span className="ml-2 flex gap-1">
                    <span className="w-1 h-1 bg-green-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1 h-1 bg-green-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1 h-1 bg-green-500 rounded-full animate-bounce" />
                  </span>
                )}
              </>
            ) : 'You'}
            <span className="text-xs opacity-50">•</span>
            <span className="text-xs flex items-center gap-1 opacity-50">
              <Clock size={12} />
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
              })}
            </span>
          </div>

          <div className={`space-y-4 text-base leading-relaxed ${
            isBot ? 'text-gray-800 dark:text-gray-200' : 'text-white'
          }`}>
            {renderContent()}
          </div>

          {/* Sources Section */}
          {!isTyping && sources.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
              <button
                onClick={() => setIsSourcesExpanded(!isSourcesExpanded)}
                className={`w-full flex items-center justify-between text-sm font-medium mb-2
                  ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}
                  hover:opacity-80 transition-opacity`}
              >
                <span className="flex items-center gap-2">
                  <ExternalLink size={16} />
                  Sources ({sources.length})
                </span>
                <ChevronDown
                  size={16}
                  className={`transform transition-transform duration-200 ${
                    isSourcesExpanded ? 'rotate-180' : ''
                  }`}
                />
              </button>
              {isSourcesExpanded && (
                <div className="space-y-2 mt-2 animate-fadeIn">
                  {sources.map((source, index) => (
                    <a
                      key={index}
                      href={source}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`flex items-center gap-2 text-sm group/link
                        ${isBot 
                          ? 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300' 
                          : 'text-white hover:text-white/90'}`}
                    >
                      <ExternalLink size={16} className="group-hover/link:translate-x-0.5 transition-transform" />
                      <span className="underline underline-offset-2">{source}</span>
                    </a>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Products Section */}
          {!isTyping && products.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
              <button
                onClick={() => setIsProductsExpanded(!isProductsExpanded)}
                className={`w-full flex items-center justify-between text-sm font-medium mb-2
                  ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}
                  hover:opacity-80 transition-opacity`}
              >
                <span className="flex items-center gap-2">
                  <ShoppingBag size={16} />
                  Recommended Products ({products.length})
                </span>
                <ChevronDown
                  size={16}
                  className={`transform transition-transform duration-200 ${
                    isProductsExpanded ? 'rotate-180' : ''
                  }`}
                />
              </button>
              {isProductsExpanded && (
                <div className="space-y-2 mt-2 animate-fadeIn">
                  {products.map((product, index) => (
                    <a
                      key={index}
                      href={product.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`flex items-center gap-2 text-sm group/link
                        ${isBot 
                          ? 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300' 
                          : 'text-white hover:text-white/90'}`}
                    >
                      <ShoppingBag size={16} className="group-hover/link:translate-x-0.5 transition-transform" />
                      <span className="underline underline-offset-2">{product.name}</span>
                    </a>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Follow-up Questions */}
          {!isTyping && isBot && moreTopic.length > 0 && (
            <div className="mt-4 pt-4 border-t border-green-100 dark:border-green-800">
              <div className="text-sm font-medium mb-3 text-green-700 dark:text-green-300">
                Would you like to know more about:
              </div>
              <div className="flex flex-wrap gap-2">
                {moreTopic.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => onFollowUpClick?.(question)}
                    className="flex items-center gap-2 text-sm bg-green-100 dark:bg-green-800/50 
                             hover:bg-green-200 dark:hover:bg-green-800 rounded-full px-4 py-2.5
                             transition-all duration-200 cursor-pointer text-green-700 dark:text-green-300
                             hover:shadow-md active:scale-95"
                  >
                    <MessageCircle size={16} className="text-green-500" />
                    <span>{question}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Attachments */}
          {message.attachments && message.attachments.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
              <div className={`text-sm font-medium mb-2
                ${isBot ? 'text-gray-700 dark:text-gray-300' : 'text-white'}`}>
                Attachments
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {message.attachments.map((file, index) => (
                  <div
                    key={index}
                    className={`flex items-start gap-3 p-3 rounded-lg group/attachment
                      ${isBot 
                        ? 'bg-gray-50 dark:bg-gray-700/50 text-gray-600 dark:text-gray-300' 
                        : 'bg-white/10 text-white'}`}
                  >
                    {isImageFile(file) ? (
                      <div className="relative group/image">
                        <div className="w-16 h-16 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-600 flex items-center justify-center">
                          <img
                            src={URL.createObjectURL(message.files?.[index] as Blob)}
                            alt={file}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxyZWN0IHg9IjMiIHk9IjMiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCIgcng9IjIiIHJ5PSIyIi8+PGNpcmNsZSBjeD0iOC41IiBjeT0iOC41IiByPSIxLjUiLz48cG9seWxpbmUgcG9pbnRzPSIyMSAxNSAxNiAxMCA1IDIxIi8+PC9zdmc+';
                            }}
                          />
                        </div>
                        <div className="hidden group-hover/image:block absolute left-20 bottom-0 z-10">
                          <img
                            src={URL.createObjectURL(message.files?.[index] as Blob)}
                            alt={file}
                            className="max-w-sm rounded-lg shadow-xl border-2 border-white dark:border-gray-700"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.style.display = 'none';
                            }}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-lg bg-gray-100 dark:bg-gray-600 flex items-center justify-center flex-shrink-0">
                        <FileText size={20} className="text-gray-500 dark:text-gray-400" />
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{file}</p>
                      {message.files?.[index] && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          {(message.files[index] as File).size > 1024 * 1024
                            ? `${((message.files[index] as File).size / (1024 * 1024)).toFixed(1)} MB`
                            : `${((message.files[index] as File).size / 1024).toFixed(1)} KB`}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
 