import { query, mutate, tx, reauthenticate } from "@onflow/fcl";
import "./testnet-config";

const getFlowBalance = async (address) => {
  const cadence = `
    import FlowToken from 0xFLOW
    import FungibleToken from 0xFT
    
    pub fun main(address: Address): UFix64{
      let account = getAccount(address)
      let path = /public/flowTokenBalance

      let vaultRef = account.getCapability(path)
        .borrow<&FlowToken.Vault{FungibleToken.Balance}>()
        ?? panic("Could not borrow Balance reference to the Vault")

      return vaultRef.balance
    }
  `;
  const args = (arg, t) => [arg(address, t.Address)];
  const balance = await query({ cadence, args });
  console.log({ balance });
};

const sendFlow = async (recepient, amount) => {
  const cadence = `
    import FungibleToken from 0xFT
    import FlowToken from 0xFLOW

    transaction(recepient: Address, amount: UFix64){
      prepare(signer: AuthAccount){
        let sender = signer.borrow<&FlowToken.Vault>(from: /storage/flowTokenVault)
          ?? panic("Could not borrow Provider reference to the Vault")

        let receiverAccount = getAccount(recepient)

        let receiver = receiverAccount.getCapability(/public/flowTokenReceiver)
          .borrow<&FlowToken.Vault{FungibleToken.Receiver}>()
          ?? panic("Could not borrow Receiver reference to the Vault")
 
        receiver.deposit(from: <- sender.withdraw(amount: amount))
      }
    }
  `;
  const args = (arg, t) => [arg(recepient, t.Address), arg(amount, t.UFix64)];
  const limit = 500;

  const txId = await mutate({
    cadence,
    args,
    limit,
  });
  console.log("Waiting for transaction to be sealed...");
  const txDetails = await tx(txId).onceSealed();
  console.log({ txDetails });
};


export const runScript = (balance) => {
  console.log("External script is running!");
  // Add your script's logic here
  (async () => {
    console.clear();
    // "reauthenticate" will ensure your session works properly
    // and present you a popup to sign in
    await reauthenticate();
  
    // This is an example account we've created to this exibition
    // You can replace it with one of your addresses
    const recepient = "0x982aeefa208cb8a7";
  
    // Check "initial" balance first
    await getFlowBalance(recepient);
  
    // Send some FLOW tokens to Recepient
    await sendFlow(recepient, balance);
  
    // Ensure that Recepient's balance has been changed
    await getFlowBalance(recepient);
  })();
};