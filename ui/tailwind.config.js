// /** @type {import('tailwindcss').Config} */
// export default {
//   content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
//   darkMode: 'class',
//   theme: {
//     extend: {
//       animation: {
//         'spin-slow': 'spin 8s linear infinite',
//         'float-delayed-1': 'float 3s ease-in-out infinite',
//         'float-delayed-2': 'float 3s ease-in-out infinite 1s',
//         'float-delayed-3': 'float 3s ease-in-out infinite 2s',
//         'float-delayed-4': 'float 3s ease-in-out infinite 3s',
//         'float-delayed-5': 'float 3s ease-in-out infinite 4s',
//         'fade-in-up': 'fadeInUp 1s ease-out forwards',
//         'scaleX': 'scaleX 0.6s ease-in-out forwards',
//         'pulse-slow': 'pulse 3s ease-in-out infinite'
//       },
//       keyframes: {
//         float: {
//           '0%, 100%': { transform: 'translateY(0)' },
//           '50%': { transform: 'translateY(-10px)' }
//         },
//         fadeInUp: {
//           '0%': { opacity: '0', transform: 'translateY(10px)' },
//           '100%': { opacity: '1', transform: 'translateY(0)' }
//         },
//         scaleX: {
//           '0%': { transform: 'scaleX(0)' },
//           '100%': { transform: 'scaleX(1)' }
//         }
//       },
//       typography: {
//         DEFAULT: {
//           css: {
//             maxWidth: 'none',
//             code: {
//               backgroundColor: '#f1f5f9',
//               padding: '0.2rem 0.4rem',
//               borderRadius: '0.25rem',
//               fontWeight: '400',
//             },
//             'code::before': {
//               content: '""'
//             },
//             'code::after': {
//               content: '""'
//             }
//           }
//         }
//       }
//     },
//   },
//   plugins: [
//     require('@tailwindcss/typography'),
//   ],
// };


/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      animation: {
        'spin-slow': 'spin 8s linear infinite',
        'float-delayed-1': 'float 3s ease-in-out infinite',
        'float-delayed-2': 'float 3s ease-in-out infinite 1s',
        'float-delayed-3': 'float 3s ease-in-out infinite 2s',
        'float-delayed-4': 'float 3s ease-in-out infinite 3s',
        'float-delayed-5': 'float 3s ease-in-out infinite 4s',
        'fade-in-up': 'fadeInUp 1s ease-out forwards',
        'bounce-slow': 'bounce 2s infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite'
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' }
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' }
        },
        bounce: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' }
        }
      },
      backgroundImage: {
        'food-pattern': "url('https://images.unsplash.com/photo-1490818387583-1baba5e638af?auto=format&fit=crop&w=1000&q=80')"
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};