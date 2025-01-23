import { useState, useEffect } from 'react';

// export function useTypewriter(text: string, speed: number = 30) {
//   const [displayedText, setDisplayedText] = useState('');
//   const [isTyping, setIsTyping] = useState(true);

//   useEffect(() => {
//     setIsTyping(true);
//     setDisplayedText('');
    
//     const words = text.split(' ');
//     let currentIndex = 0;
//     const intervalId = setInterval(() => {
//       if (currentIndex < words.length-1) {
//         setDisplayedText(prev => prev + (prev ? ' ' : '') + words[currentIndex]);
//         currentIndex++;
//       } else {
//         setIsTyping(false);
//         clearInterval(intervalId);
//       }
//     }, speed);

//     return () => clearInterval(intervalId);
//   }, [text, speed]);

//   return { displayedText, isTyping };
// }


export function useTypewriter(text: string, speed: number = 30) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    setIsTyping(true);
    setDisplayedText('');
    
    const words = text.split(' ');
    let currentIndex = 0;
    
    const intervalId = setInterval(() => {
      if (currentIndex < words.length) {
        // Simpler approach: Join all words up to current index with spaces
        const newText = words.slice(0, currentIndex + 1).join(' ');
        setDisplayedText(newText);
        currentIndex++;
      } else {
        setIsTyping(false);
        clearInterval(intervalId);
      }
    }, speed);

    return () => clearInterval(intervalId);
  }, [text, speed]);

  return { displayedText, isTyping };
}