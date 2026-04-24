import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Language Learning Assistant",
  description: "Learn Mandarin Chinese through the Hán Việt bridge — leveraging the 60% Sino-Vietnamese vocabulary overlap between Vietnamese and Chinese.",
  keywords: ["Mandarin", "Chinese", "Vietnamese", "Hán Việt", "Sino-Vietnamese", "language learning", "tutor"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-gradient-to-br from-violet-50 via-white to-indigo-50">
        {children}
      </body>
    </html>
  );
}
