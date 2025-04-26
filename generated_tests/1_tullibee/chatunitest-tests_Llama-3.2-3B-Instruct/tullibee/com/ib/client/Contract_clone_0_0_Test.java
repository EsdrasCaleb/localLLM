package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Contract_clone_0_0_Test {

    @Test
    public void testClone() throws CloneNotSupportedException {
        Contract contract = new Contract(1, "ABC", "FUT", "2022-01-01", 100.0, "C", "S", "NYMEX", "USD", "ABC", new Vector(), "ABC", false, "STK", "123456789");
        Contract clonedContract = (Contract) contract.clone();
        // Verify that the cloned contract is a deep copy
        assertNotSame(contract, clonedContract);
        // Verify that the cloned contract has the same properties as the original contract
        assertEquals(contract.m_conId, clonedContract.m_conId);
        assertEquals(contract.m_symbol, clonedContract.m_symbol);
        assertEquals(contract.m_secType, clonedContract.m_secType);
        assertEquals(contract.m_expiry, clonedContract.m_expiry);
        assertEquals(contract.m_strike, clonedContract.m_strike);
        assertEquals(contract.m_right, clonedContract.m_right);
        assertEquals(contract.m_multiplier, clonedContract.m_multiplier);
        assertEquals(contract.m_exchange, clonedContract.m_exchange);
        assertEquals(contract.m_currency, clonedContract.m_currency);
        assertEquals(contract.m_localSymbol, clonedContract.m_localSymbol);
        assertEquals(contract.m_primaryExch, clonedContract.m_primaryExch);
        assertEquals(contract.m_includeExpired, clonedContract.m_includeExpired);
        assertEquals(contract.m_secIdType, clonedContract.m_secIdType);
        assertEquals(contract.m_secId, clonedContract.m_secId);
        assertEquals(contract.m_comboLegsDescrip, clonedContract.m_comboLegsDescrip);
        assertEquals(contract.m_comboLegs, clonedContract.m_comboLegs);
        // Verify that the cloned contract has its own Vector
        assertNotSame(contract.m_comboLegs, clonedContract.m_comboLegs);
        // Verify that the cloned contract has its own UnderComp object
        assertNotSame(contract.m_underComp, clonedContract.m_underComp);
        // Verify that the cloned contract is not the same as the original contract
        assertNotSame(contract, clonedContract);
    }
}
