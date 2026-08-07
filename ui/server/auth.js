/**
 * Cognito JWT verification for the BADGERS UI server.
 *
 * Aligned with the media-contracts (MC) reference implementation: the browser
 * performs an OIDC authorization-code + PKCE flow against the Cognito hosted UI
 * and sends the resulting access token as `Authorization: Bearer <token>`. This
 * module verifies that token against the user pool's JWKS.
 *
 * This replaces the previous ALB-injected `x-amzn-oidc-data` verification, which
 * only worked behind a self-managed Application Load Balancer configured with an
 * authenticate-cognito action. The ECS Express Gateway service does not inject
 * that header, so identity is now established at the application layer.
 *
 * Local development: when COGNITO_USER_POOL_ID is unset, auth is bypassed and a
 * dev identity is used. BADGERS_UI_ROLE selects the role for that identity.
 */

import { createRemoteJWKSet, jwtVerify } from 'jose';

const IS_ECS = !!process.env.ECS_CONTAINER_METADATA_URI_V4;
let _jwks = null;
let _localDevLogged = false;

function cfg() {
    return {
        userPoolId: process.env.COGNITO_USER_POOL_ID || '',
        clientId: process.env.COGNITO_UI_CLIENT_ID || '',
        region: process.env.AWS_REGION || 'us-west-2',
    };
}

/**
 * Auth is bypassed only when no user pool is configured AND we are not running
 * on ECS. On ECS a missing user pool is a misconfiguration, not a dev shortcut.
 */
function isLocalDev() {
    const { userPoolId } = cfg();
    const val = !userPoolId && !IS_ECS;
    if (val && !_localDevLogged) {
        console.warn(
            '[auth] COGNITO_USER_POOL_ID is not set and this is not ECS — ' +
            'bypassing JWT verification. Do NOT use in production.'
        );
        _localDevLogged = true;
    }
    return val;
}

function getJwks() {
    const { userPoolId, region } = cfg();
    if (!_jwks && userPoolId) {
        const jwksUri =
            `https://cognito-idp.${region}.amazonaws.com/${userPoolId}/.well-known/jwks.json`;
        _jwks = createRemoteJWKSet(new URL(jwksUri));
    }
    return _jwks;
}

function localDevUser() {
    return {
        email: 'local-dev',
        name: 'Local Dev',
        sub: 'local',
        groups: [process.env.BADGERS_UI_ROLE || 'admin'],
        verified: false,
    };
}

/**
 * Verify the bearer token and return a normalised identity, or null.
 */
export async function getUser(req) {
    if (isLocalDev()) return localDevUser();

    const { userPoolId, clientId, region } = cfg();
    if (!userPoolId) return null;

    const header = req.headers.authorization || '';
    const token = header.startsWith('Bearer ') ? header.slice(7) : null;
    if (!token) return null;

    try {
        const { payload } = await jwtVerify(token, getJwks(), {
            issuer: `https://cognito-idp.${region}.amazonaws.com/${userPoolId}`,
        });

        // Access tokens carry client_id; ID tokens carry aud. Either must match.
        if (clientId) {
            const tokenClient = payload.client_id || payload.aud;
            if (tokenClient !== clientId) {
                console.warn('[auth] token client mismatch:', tokenClient);
                return null;
            }
        }

        return {
            email: payload.email || payload.username || payload['cognito:username'] || 'unknown',
            name: payload.name || '',
            sub: payload.sub || '',
            groups: payload['cognito:groups'] || [],
            verified: true,
        };
    } catch (e) {
        console.warn('[auth] token verification failed:', e.message);
        return null;
    }
}

/** Express middleware — 401 unless a valid token is present. */
export function requireAuth(req, res, next) {
    getUser(req)
        .then((user) => {
            if (!user) return res.status(401).json({ error: 'Authentication required' });
            req.user = user;
            next();
        })
        .catch((err) => {
            console.error('[auth] middleware error:', err.message);
            res.status(500).json({ error: 'Authentication error' });
        });
}

/** Express middleware — 403 unless the caller is in the admin group. */
export function requireAdmin(req, res, next) {
    getUser(req)
        .then((user) => {
            if (!user) return res.status(401).json({ error: 'Authentication required' });
            if (!user.groups.includes('admin')) {
                return res.status(403).json({ error: 'Admin access required' });
            }
            req.user = user;
            next();
        })
        .catch((err) => {
            console.error('[auth] middleware error:', err.message);
            res.status(500).json({ error: 'Authentication error' });
        });
}
