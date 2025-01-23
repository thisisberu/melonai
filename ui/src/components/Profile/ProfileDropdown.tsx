import React from 'react';
import { User, Mail, Phone, Calendar } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

interface ProfileDropdownProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ProfileDropdown2({ isOpen, onClose }: ProfileDropdownProps) {
  const { user } = useAuth();
  console.log("user deyaols",user); 
  if (!isOpen) return null;

  return (
    <div className="absolute right-0 mt-2 w-80 rounded-lg bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black ring-opacity-5 z-50">
      <div className="p-4">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white">
            <User size={24} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {user?.aboId}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">ABO Member</p>
          </div>
        </div>

        {/* <div className="space-y-3">
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <User size={18} />
            <span>ABO ID: {user?.aboId}</span>
          </div>
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <Calendar size={18} />
            <span>Joined: {new Date(user?.createdAt || '').toLocaleDateString()}</span>
          </div>
        </div> */}

<div className="space-y-3">
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <User size={18} />
            <span>Name: {user?.full_name || 'N/A'}</span>
          </div>
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <Mail size={18} />
            <span>Email: {user?.email || 'N/A'}</span>
          </div>
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <Phone size={18} />
            <span>Phone: {user?.mobile_number || 'N/A'}</span>
          </div>
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <User size={18} />
            <span>ABO ID: {user?.aboId}</span>
          </div>
          <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
            <Calendar size={18} />
            <span>Joined: {new Date(user?.createdAt || '').toLocaleDateString()}</span>
          </div>
        </div>
        

        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onClose}
            className="w-full text-center text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export function ProfileDropdown({ isOpen, onClose }: ProfileDropdownProps) {
  const { user } = useAuth();
  // console.log("tyep of profile pic", typeof(user?.profile_picture))
  if (!isOpen) return null;

  return (
    <div className="absolute right-0 mt-2 w-80 rounded-lg bg-white dark:bg-gray-800 shadow-lg ring-1 ring-black ring-opacity-5 z-50">
      <div className="p-4">
        <div className="flex items-center space-x-3 mb-4">
          {user?.profile_picture ? (
            <img
              src={typeof user.profile_picture === 'string' 
                ? user.profile_picture 
                : URL.createObjectURL(user.profile_picture)}
              // src={user.profile_picture}
              alt="Profile"
              className="w-12 h-12 rounded-full object-cover"
            />
          ) : (
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white">
              <User size={24} />
            </div>
          )}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {user?.full_name || user?.aboId || 'N/A'}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">ABO Member</p>
          </div>
        </div>

        <div className="space-y-3">
          {user?.full_name && (
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <User size={18} />
              <span>Name: {user.full_name}</span>
            </div>
          )}
          {user?.email && (
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Mail size={18} />
              <span>Email: {user.email}</span>
            </div>
          )}
          {user?.mobile_number && (
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Phone size={18} />
              <span>Phone: {user.mobile_number}</span>
            </div>
          )}
          {user?.aboId && (
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <User size={18} />
              <span>ABO ID: {user.aboId}</span>
            </div>
          )}
          {user?.createdAt && (
            <div className="flex items-center space-x-3 text-gray-700 dark:text-gray-300">
              <Calendar size={18} />
              <span>Joined: {new Date(user.createdAt).toLocaleDateString()}</span>
            </div>
          )}
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onClose}
            className="w-full text-center text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

