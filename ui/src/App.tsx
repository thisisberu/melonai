

// import React from 'react';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
// import { AuthProvider } from './hooks/useAuth';
// import { LoginForm } from './components/auth/LoginForm';
// import { RegisterForm } from './components/auth/RegisterForm';
// import { Chat } from './components/Chat/index';
// import { PrivateRoute } from './components/auth/PrivateRoute';

// const queryClient = new QueryClient();

// function App() {
//   return (
//     <QueryClientProvider client={queryClient}>
//       <AuthProvider>
//         <Router>
//           <Routes>
//             <Route path="/login" element={<LoginForm />} />
//             <Route path="/register" element={<RegisterForm />} />
//             <Route
//               path="/chat"
//               element={
//                 <PrivateRoute>
//                   <Chat />
//                 </PrivateRoute>
//               }
//             />
//             <Route path="/" element={<Navigate to="/login" replace />} />
//           </Routes>
//         </Router>
//       </AuthProvider>
//     </QueryClientProvider>
//   );
// }

// export default App;

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './hooks/useAuth';
import { LoginForm } from './components/auth/LoginForm';
import { RegisterForm } from './components/auth/RegisterForm';
import { RegistrationChat } from './components/auth/RegistrationChat';
import { Chat } from './components/Chat';
import { PrivateRoute } from './components/auth/PrivateRoute';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<LoginForm />} />
            <Route path="/register" element={<RegisterForm />} />
            <Route path="/register-chat" element={<RegistrationChat />} />
            <Route
              path="/chat"
              element={
                <PrivateRoute>
                  <Chat />
                </PrivateRoute>
              }
            />
            <Route path="/" element={<Navigate to="/login" replace />} />
          </Routes>
        </Router>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;