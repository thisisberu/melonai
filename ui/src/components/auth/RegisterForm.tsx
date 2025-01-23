import React, { useId, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { WellnessLogo } from '../WellnessLogo';
import type { RegisterCredentials } from '../../types/auth';

import { useLocation } from 'react-router-dom';

interface LocationState {
  userInfo: {
    // name: string;
    // email: string;
    // phone: string;
    // purpose: string;

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
}


export function RegisterForm() {
  const location = useLocation();
  const { userInfo } = (location.state as LocationState) || { userInfo: null };
  console.log("🚀 ~ RegisterForm ~ userInfo:", userInfo)

  const [credentials, setCredentials] = useState<RegisterCredentials>({
    aboId: '',
    password: '',
    confirmPassword: '',
    // name: userInfo.name,
    // email:userInfo.email,
    // phone_number: userInfo.phone,

    // Personal Details
    full_name: userInfo.full_name,
    date_of_birth: userInfo.date_of_birth,
    gender: userInfo.gender,
    ethnicity: userInfo.ethnicity,
    profile_picture: userInfo.profile_picture,
    
    // Communication Details
    mobile_number: userInfo.mobile_number,
    email: userInfo.email,
    address: userInfo.address,
    
    // Identity Details
    national_id: userInfo.national_id,
    identity_proof: userInfo.identity_proof,
    
    // Sponsor Details
    sponsor_abo_id: userInfo.sponsor_abo_id,
    sponsor_abo_name: userInfo.sponsor_abo_name,
    
    // Payment Details
    card_number: userInfo.card_number,
    card_name: userInfo.card_name,
    
    // Final Verification
    id_proof_image: userInfo.id_proof_image

  }); 
  console.log("🚀 ~ RegisterForm ~ credentials:", credentials)
  
  // console.log("Inside register",userInfo); 

  const [error, setError] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (credentials.password !== credentials.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setIsLoading(true);

    try {
      await register(credentials);
      navigate('/chat');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to register');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <WellnessLogo />
        </div>

        <div className="bg-white dark:bg-gray-800 py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {error && (
              <div className="rounded-md bg-red-50 dark:bg-red-900/50 p-4">
                <div className="text-sm text-red-700 dark:text-red-200">{error}</div>
              </div>
            )}

            <div>
              <label htmlFor="abo-id" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                ABO ID
              </label>
              <div className="mt-1">
                <input
                  id="abo-id"
                  name="aboId"
                  type="text"
                  required
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
                  value={credentials.aboId}
                  onChange={(e) => setCredentials({ ...credentials, aboId: e.target.value })}
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Password
              </label>
              <div className="mt-1">
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
                  value={credentials.password}
                  onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                />
              </div>
            </div>

            <div>
              <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Confirm Password
              </label>
              <div className="mt-1">
                <input
                  id="confirm-password"
                  name="confirmPassword"
                  type="password"
                  required
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
                  value={credentials.confirmPassword}
                  onChange={(e) => setCredentials({ ...credentials, confirmPassword: e.target.value })}
                />
              </div>
            </div>

            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Creating account...' : 'Create account'}
              </button>
            </div>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300 dark:border-gray-600" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white dark:bg-gray-800 text-gray-500">
                  Already have an account?
                </span>
              </div>
            </div>

            <div className="mt-6">
              <Link
                to="/login"
                className="w-full flex justify-center py-2 px-4 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Sign in instead
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}