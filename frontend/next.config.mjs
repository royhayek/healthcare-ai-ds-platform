/** @type {import('next').NextConfig} */
const nextConfig = {
  // FASTAPI_URL is consumed by the server-side proxy route only.
  // It must NOT be prefixed with NEXT_PUBLIC_ - the backend URL should
  // never be exposed to the browser.
  env: {
    FASTAPI_URL: "http://127.0.0.1:8001",
  },
  // react-markdown and its remark/rehype ecosystem are ESM-only packages.
  // Next.js requires them in transpilePackages to bundle correctly.
  transpilePackages: [
    "react-markdown",
    "remark-gfm",
    "remark-parse",
    "remark-rehype",
    "rehype-stringify",
    "unified",
    "bail",
    "is-plain-obj",
    "trough",
    "vfile",
    "vfile-message",
    "unist-util-stringify-position",
    "mdast-util-from-markdown",
    "mdast-util-to-markdown",
    "mdast-util-gfm",
    "micromark",
    "micromark-core-commonmark",
    "micromark-extension-gfm",
    "micromark-util-character",
    "micromark-util-chunked",
    "micromark-util-classify-character",
    "micromark-util-combine-extensions",
    "micromark-util-decode-numeric-character-reference",
    "micromark-util-decode-string",
    "micromark-util-encode",
    "micromark-util-html-tag-name",
    "micromark-util-normalize-identifier",
    "micromark-util-resolve-all",
    "micromark-util-sanitize-uri",
    "micromark-util-subtokenize",
    "micromark-util-symbol",
    "micromark-util-types",
    "devlop",
  ],
};

export default nextConfig;
