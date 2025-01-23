// import React from 'react';
// import { Moon, Sun } from 'lucide-react';
// import { useTheme } from '../hooks/useTheme';

// export function ThemeToggle() {
//   const { theme, toggleTheme } = useTheme();

//   return (
//     <button
//       onClick={toggleTheme}
//       className="p-2 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
//       title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
//     >
//       {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
//     </button>
//   );
// }


import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="p-2 text-pink-500 hover:text-pink-700 dark:text-pink-400 dark:hover:text-pink-300
               hover:bg-pink-100 dark:hover:bg-pink-800/50 rounded-lg transition-all duration-200
               transform hover:scale-105"
      title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  );
}