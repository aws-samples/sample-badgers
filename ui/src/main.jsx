import React from 'react'
import ReactDOM from 'react-dom/client'
import { AuthProvider } from 'react-oidc-context'
import { UserProvider } from './hooks/useUser.js'
import App from './App.jsx'
import './index.css'

// Cognito OIDC (authorization code + PKCE). Values are baked into the Vite
// bundle at build time by deployment/scripts/generate_ui_env.sh.
const cognitoAuthConfig = {
    authority: import.meta.env.VITE_COGNITO_AUTHORITY,
    client_id: import.meta.env.VITE_COGNITO_CLIENT_ID,
    redirect_uri: `${window.location.origin}/callback`,
    response_type: 'code',
    scope: 'openid email profile',
    automaticSilentRenew: true,
    onSigninCallback: () => {
        window.history.replaceState({}, document.title, '/')
    },
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <AuthProvider {...cognitoAuthConfig}>
        <UserProvider>
            <App />
        </UserProvider>
    </AuthProvider>
)
