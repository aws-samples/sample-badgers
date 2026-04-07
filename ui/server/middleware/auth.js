import jwt from 'jsonwebtoken';

const LOCAL_ROLE = process.env.BADGERS_UI_ROLE || 'admin';
const keyCache = new Map();

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
 * Falls back to local dev defaults when no OIDC header is present.
 */
export async function getUserFromOIDC(req) {
    const token = req.headers['x-amzn-oidc-data'];
    if (!token) {
        return {
            email: 'local-dev',
            name: 'Local Dev',
            sub: 'local',
            groups: [LOCAL_ROLE],
            verified: false,
        };
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
        return {
            email: 'unknown',
            name: '',
            sub: '',
            groups: [],
            verified: false,
        };
    }
}

/**
 * Express middleware that blocks non-admin users with 403.
 */
export function requireAdmin(req, res, next) {
    getUserFromOIDC(req).then(user => {
        if (user.groups.includes('admin')) {
            req.user = user;
            return next();
        }
        res.status(403).json({ error: 'Admin access required' });
    }).catch(err => {
        console.error('[auth] middleware error:', err.message);
        res.status(500).json({ error: 'Authentication error' });
    });
}
