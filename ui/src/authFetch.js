// Authenticated fetch helpers.
//
// Two mechanisms are provided:
//
//  1. `createAuthFetch(accessToken)` — an explicit wrapper, matching the
//     media-contracts reference. Use it when a component already receives an
//     access token as a prop.
//
//  2. `installAuthFetch(getAccessToken)` — patches the global `fetch` so that
//     same-origin requests to `/api/*` automatically carry the bearer token.
//     The server enforces auth on every `/api/*` route, and the UI has 34
//     separate fetch call sites; installing a single interceptor guarantees
//     none of them can silently regress to an unauthenticated request.
//     Requests that already set an Authorization header are left untouched.

export function createAuthFetch(accessToken) {
    return async function authFetch(url, options = {}) {
        const headers = {
            ...options.headers,
            ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        };
        return fetch(url, { ...options, headers });
    };
}

let _installed = false;

/**
 * @param {() => string | undefined} getAccessToken resolves the current token
 */
export function installAuthFetch(getAccessToken) {
    if (_installed) return;
    _installed = true;

    const nativeFetch = window.fetch.bind(window);

    window.fetch = async (input, init = {}) => {
        let path = '';
        try {
            if (typeof input === 'string') {
                path = new URL(input, window.location.origin).pathname;
            } else if (input instanceof Request) {
                path = new URL(input.url, window.location.origin).pathname;
            } else if (input instanceof URL) {
                path = input.pathname;
            }
        } catch {
            path = '';
        }

        if (!path.startsWith('/api/')) return nativeFetch(input, init);

        const token = getAccessToken();
        if (!token) return nativeFetch(input, init);

        // Respect an explicitly supplied Authorization header.
        const existing = new Headers(
            init.headers || (input instanceof Request ? input.headers : undefined)
        );
        if (existing.has('Authorization')) return nativeFetch(input, init);
        existing.set('Authorization', `Bearer ${token}`);

        if (input instanceof Request) {
            return nativeFetch(new Request(input, { headers: existing }), init);
        }
        return nativeFetch(input, { ...init, headers: existing });
    };
}
