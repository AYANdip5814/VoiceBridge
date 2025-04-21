/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: "#1f1b3a",
                accent: "#6c5ce7"
            }
        },
    },
    plugins: [],
}
