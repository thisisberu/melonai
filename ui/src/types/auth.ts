export interface User {
    id: number;
    aboId: string;
    createdAt: Date;
    chat_number: number;
    // name: string,
    // email:string, 
    // phone: string

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
  
  export interface LoginCredentials {
    aboId: string;
    password: string;
  }
  
  export interface RegisterCredentials extends LoginCredentials {
    confirmPassword: string;
    // name: string, 
    // email: string,
    // phone_number: string
    // user_details: User

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
  
  export interface AuthResponse {
    user: User;
    token: string;
  }