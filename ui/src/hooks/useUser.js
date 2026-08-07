/**
 * Current-user context for the BADGERS UI.
 *
 * Identity comes from the ID token the OIDC authorization-code + PKCE flow already
 * produced, read via react-oidc-context. There is no /api/me round trip: the token is
 * validated by oidc-client-ts during the flow (signature, issuer, audience, nonce), and
 * asking the server to echo back claims the browser already holds added a request and an
 * endpoint without adding information.
 *
 * The role here decides which tabs render, nothing more. It is not an authorization
 * boundary — every admin route enforces requireAdmin server-side (ui/server/routes/admin.js),
 * so a client that claims to be admin still gets 403 from the API.
 *
 * Local development: with no Cognito configured there is no token, matching the server's
 * own bypass in ui/server/auth.js. Role then falls back to VITE_BADGERS_UI_ROLE, or admin
 * when that is unset, which mirrors the previous BADGERS_UI_ROLE default.
 */

import React, { createContext, useContext, useMemo } from 'react';
import { useAuth } from 'react-oidc-context';
import { installAuthFetch } from '../authFetch.js';

// Auth is only enforced when the bundle was built with Cognito config, matching
// AUTH_ENABLED in App.jsx.
const AUTH_ENABLED = Boolean(
    import.meta.env.VITE_COGNITO_AUTHORITY && import.meta.env.VITE_COGNITO_CLIENT_ID
);
const DEV_ROLE = import.meta.env.VITE_BADGERS_UI_ROLE || 'admin';

const UserContext = createContext({
    email: '',
    name: '',
    role: 'tester',
    verified: false,
    loading: true,
    accessToken: undefined,
});

export function UserProvider({ children }) {
    const auth = useAuth();
    const accessToken = auth.user?.access_token;

    // Install the global fetch interceptor once, reading the token lazily so it always
    // picks up the current value (including after a silent renew).
    const tokenRef = React.useRef(accessToken);
    tokenRef.current = accessToken;
    React.useEffect(() => {
        installAuthFetch(() => tokenRef.current);
    }, []);

    const value = useMemo(() => {
        if (!AUTH_ENABLED) {
            return {
                email: 'local-dev',
                name: 'Local Dev',
                role: DEV_ROLE,
                verified: false,
                loading: false,
                accessToken,
            };
        }

        // Wait for the OIDC flow to settle before reporting an identity, so consumers do
        // not briefly render as an unprivileged user mid-redirect.
        if (auth.isLoading) {
            return { email: '', name: '', role: 'tester', verified: false, loading: true, accessToken };
        }

        const claims = auth.user?.profile || {};
        const groups = claims['cognito:groups'] || [];
        return {
            email: claims.email || claims['cognito:username'] || '',
            name: claims.name || '',
            role: groups.includes('admin') ? 'admin' : 'tester',
            verified: Boolean(auth.isAuthenticated),
            loading: false,
            accessToken,
        };
    }, [auth.isLoading, auth.isAuthenticated, auth.user, accessToken]);

    return React.createElement(UserContext.Provider, { value }, children);
}

export function useUser() {
    return useContext(UserContext);
}
