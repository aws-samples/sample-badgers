import React, { createContext, useContext, useState, useEffect } from 'react';

const UserContext = createContext({ email: '', name: '', role: 'tester', verified: false, loading: true });

export function UserProvider({ children }) {
    const [user, setUser] = useState({ email: '', name: '', role: 'tester', verified: false, loading: true });

    useEffect(() => {
        fetch('/api/me')
            .then(r => r.json())
            .then(data => setUser({ ...data, loading: false }))
            .catch(() => setUser(prev => ({ ...prev, loading: false })));
    }, []);

    return React.createElement(UserContext.Provider, { value: user }, children);
}

export function useUser() {
    return useContext(UserContext);
}
