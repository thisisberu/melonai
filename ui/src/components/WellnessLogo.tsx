// import React from 'react';

// export function WellnessLogo() {
//   return (
//     <div className="flex flex-col items-center mt-8">
//       <div className="relative w-48 h-48">
//         {/* Outer glowing ring */}
//         <div className="absolute inset-0 animate-pulse-slow opacity-20">
//           <div className="w-full h-full rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500"></div>
//         </div>

//         {/* Rotating particles */}
//         <div className="absolute inset-0 animate-spin-slow">
//           <svg viewBox="0 0 100 100" className="w-full h-full">
//             <defs>
//               <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
//                 <stop offset="0%" style={{ stopColor: '#3B82F6', stopOpacity: 0.2 }} />
//                 <stop offset="100%" style={{ stopColor: '#8B5CF6', stopOpacity: 0.2 }} />
//               </linearGradient>
//             </defs>
//             {[...Array(12)].map((_, i) => (
//               <circle
//                 key={i}
//                 cx={50 + 40 * Math.cos((i * 2 * Math.PI) / 12)}
//                 cy={50 + 40 * Math.sin((i * 2 * Math.PI) / 12)}
//                 r="2"
//                 fill="url(#grad1)"
//                 className="animate-pulse"
//               />
//             ))}
//           </svg>
//         </div>

//         {/* Main Logo Container */}
//         <div className="absolute inset-0 flex items-center justify-center">
//           <div className="relative transform transition-all duration-500 hover:scale-105">
//             {/* Logo Text */}
//             <div className="relative">
//               {/* "AMWAY" text */}
//               <div className="flex items-center justify-center">
//                 <span className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
//                   Eats
//                 </span>
//                 {/* MWAY */}
//                 {/* <span className="text-3xl font-semibold text-gray-700 dark:text-gray-300">MW</span>
//                 <span className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-blue-600">
//                   A
//                 </span>
//                 <span className="text-3xl font-semibold text-gray-700 dark:text-gray-300">Y</span> */}
//               </div>

//               {/* "INTELLIGENCE" text */}
//               <div className="text-sm font-medium text-gray-500 dark:text-gray-400 tracking-wider text-center mt-1">
//                 <span className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
//                   AI
//                 </span>
//                 {/* <span>NTELLIGENCE</span> */}
//               </div>

//               {/* Highlight effect for AI */}
//               <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-purple-500 opacity-20 blur-lg rounded-full transform scale-110"></div>
//             </div>
//           </div>
//         </div>

//         {/* Floating particles */}
//         <div className="absolute inset-0">
//           {[...Array(5)].map((_, i) => (
//             <div
//               key={i}
//               className={`absolute w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 opacity-20
//                 animate-float-delayed-${i + 1}`}
//               style={{
//                 left: `${Math.random() * 100}%`,
//                 top: `${Math.random() * 100}%`,
//                 animationDelay: `${i * 0.5}s`
//               }}
//             />
//           ))}
//         </div>
//       </div>

//       {/* Tagline */}
//       <div className="mt-6 text-gray-600 dark:text-gray-400 text-center max-w-sm">
//         <p className="animate-fade-in-up text-lg font-medium">
//           Your AI-Powered Health & Wellness Guide
//         </p>
//       </div>
//     </div>
//   );
// }


import React from 'react';
import { Apple, Carrot, Banana } from 'lucide-react';

export function WellnessLogo() {
  return (
    <div className="flex flex-col items-center mt-8">
      <div className="relative w-48 h-48">
        {/* Animated fruit background */}
        <div className="absolute inset-0">
          {[...Array(8)].map((_, i) => (
            <div
              key={i}
              className={`absolute w-8 h-8 rounded-full bg-gradient-to-br 
                ${i % 3 === 0 ? 'from-red-400 to-red-500' : // Apple colors
                i % 3 === 1 ? 'from-orange-400 to-orange-500' : // Orange colors
                'from-green-400 to-green-500'} // Leafy greens
                opacity-20 animate-float-delayed-${i + 1}`}
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${i * 0.5}s`
              }}
            />
          ))}
        </div>

        {/* Main Logo Container */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative transform transition-all duration-500 hover:scale-105">
            {/* Logo Icons */}
            <div className="flex items-center gap-4 mb-4">
              <Apple className="w-8 h-8 text-red-500 animate-bounce-slow" />
              <Carrot className="w-8 h-8 text-orange-500 animate-float-delayed-2" />
              <Banana className="w-8 h-8 text-yellow-500 animate-float-delayed-3" />
            </div>

            {/* Logo Text */}
            <div className="relative">
              <div className="flex items-center justify-center">
                <span className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-600 to-emerald-600">
                  Nutri
                </span>
                <span className="text-3xl font-semibold text-orange-500">Chat</span>
              </div>

              {/* Subtitle */}
              <div className="text-sm font-medium text-gray-500 dark:text-gray-400 tracking-wider text-center mt-1">
                Your Wellness Guide
              </div>

              {/* Glow effect */}
              <div className="absolute -inset-1 bg-gradient-to-r from-green-500 to-emerald-500 opacity-20 blur-lg rounded-full transform scale-110"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Tagline */}
      <div className="mt-6 text-gray-600 dark:text-gray-400 text-center max-w-sm">
        <p className="animate-fade-in-up text-lg font-medium">
          Nourishing Your Health Journey 🥗
        </p>
      </div>
    </div>
  );
}