import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from 'react-oidc-context';
import { installAuthFetch } from '../authFetch.js';

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

    const [user, setUser] = useState({
        email: '',
        name: '',
        role: 'tester',
        verified: false,
        loading: true,
    });

    // Install the global fetch interceptor once, reading the token lazily so it
    // always picks up the current value (including after a silent renew).
    const tokenRef = React.useRef(accessToken);
    tokenRef.current = accessToken;
    useEffect(() => {
        installAuthFetch(() => tokenRef.current);
    }, []);

    useEffect(() => {
        // Wait for the OIDC flow to settle. When Cognito is not configured
        // (local dev), isAuthenticated stays false and the server bypasses auth,
        // so /api/me is still queried and returns the dev identity.
        if (auth.isLoading) return;

        let cancelled = false;
        fetch('/api/me')
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
            .then((data) => {
                if (!cancelled) setUser({ ...data, loading: false });
            })
            .catch(() => {
                if (!cancelled) setUser((prev) => ({ ...prev, loading: false }));
            });

        return () => {
            cancelled = true;
        };
    }, [auth.isLoading, auth.isAuthenticated, accessToken]);

    return React.createElement(
        UserContext.Provider,
        { value: { ...user, accessToken } },
        children
    );
}

export function useUser() {
    return useContext(UserContext);
}
