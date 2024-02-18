'use client';

import React from 'react';

const handleButtonClick = async() => {
  console.log("button clicked, calling transaction.js");
  const script = await import('../flow/transaction');
  script.runScript("5.0");
};

export default function Home() {
  return (
    <div className="flex h-screen justify-center items-center">
      <button onClick={handleButtonClick} className="bg-blue-500 text-white font-bold py-2 px-4 rounded hover:bg-blue-700">Transact Money</button>
    </div>
  );
}
