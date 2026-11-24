/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#172c2b',
        teal: '#3f7770',
        moss: '#b7c9ad',
        paper: '#f7f7f3',
        charcoal: '#283432',
        line: '#dfe4dc',
      },
      fontFamily: {
        sans: ['Manrope', 'Avenir Next', 'Segoe UI', 'sans-serif'],
      },
      spacing: {
        1: '4px',
        2: '8px',
        3: '12px',
        4: '16px',
        5: '24px',
        6: '32px',
        7: '48px',
        8: '64px',
        9: '96px',
      },
      maxWidth: { shell: '1240px' },
    },
  },
  plugins: [],
}

