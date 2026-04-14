import jwt from 'jsonwebtoken';

const IS_ECS = !!process.env.ECS_CONTAINER_METADATA_URI_V4;
const LOCAL_DEV = process.env.BADGERS_LOCAL_DEV === 'true' && !IS_ECS;
const LOCAL_ROLE = process.env.BADGERS_UI_ROLE || 'admin';
const keyCache = new Map();

if (LOCAL_DEV) console.warn('[auth] ⚠️  BADGERS_LOCAL_DEV=true — bypassing OIDC auth. Do NOT use in production.');

async function fetchALBPublicKey(kid, region) {
    if (keyCache.has(kid)) return keyCache.get(kid);
    const url = `https://public-keys.auth.elb.${region}.amazonaws.com/${kid}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ALB public key: ${resp.status}`);
    const key = await resp.text();
    keyCache.set(kid, key);
    return key;
}

/**
 * Extract and verify user identity from ALB OIDC JWT.
 * In production (no BADGERS_LOCAL_DEV), missing OIDC header = unauthenticated.
 * Locally (BADGERS_LOCAL_DEV=true), falls back to a dev identity.
 */
export async function getUserFromOIDC(req) {
    const token = req.headers['x-amzn-oidc-data'];
    if (!token) {
        if (LOCAL_DEV) {
            return {
                email: 'local-dev',
                name: 'Local Dev',
                sub: 'local',
                groups: [LOCAL_ROLE],
                verified: false,
            };
        }
        return null;
    }

    const region = process.env.AWS_REGION || 'us-west-2';
    try {
        const decoded = jwt.decode(token, { complete: true });
        if (!decoded?.header?.kid) throw new Error('Missing kid in JWT header');

        const pubKey = await fetchALBPublicKey(decoded.header.kid, region);
        const payload = jwt.verify(token, pubKey, { algorithms: ['ES256'] });

        return {
            email: payload.email || 'unknown',
            name: payload.name || '',
            sub: payload.sub || '',
            groups: payload['cognito:groups'] || [],
            verified: true,
        };
    } catch (e) {
        console.warn('[auth] Failed to verify OIDC JWT:', e.message);
        return null;
    }
}

/**
 * Express middleware that blocks unauthenticated users with 401.
 */
export function requireAuth(req, res, next) {
    getUserFromOIDC(req).then(user => {
        if (!user || !user.email || user.email === 'unknown') {
            return res.status(401).json({ error: 'Authentication required' });
        }
        req.user = user;
        next();
    }).catch(err => {
        console.error('[auth] middleware error:', err.message);
        res.status(500).json({ error: 'Authentication error' });
    });
}

/**
 * Express middleware that blocks non-admin users with 403.
 */
export function requireAdmin(req, res, next) {
    getUserFromOIDC(req).then(user => {
        if (!user) return res.status(401).json({ error: 'Authentication required' });
        if (!user.groups.includes('admin')) return res.status(403).json({ error: 'Admin access required' });
        req.user = user;
        next();
    }).catch(err => {
        console.error('[auth] middleware error:', err.message);
        res.status(500).json({ error: 'Authentication error' });
    });
}
