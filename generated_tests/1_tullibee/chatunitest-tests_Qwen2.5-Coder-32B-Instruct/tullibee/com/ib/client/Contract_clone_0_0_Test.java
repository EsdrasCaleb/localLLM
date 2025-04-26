package com.ib.client;

import java.lang.reflect.Field;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Contract_clone_0_0_Test {

    private Contract contract;

    private Vector<Object> comboLegs;

    @BeforeEach
    public void setUp() throws Exception {
        contract = new Contract();
        comboLegs = new Vector<>();
        comboLegs.add("Leg1");
        comboLegs.add("Leg2");
        Field comboLegsField = Contract.class.getDeclaredField("m_comboLegs");
        comboLegsField.setAccessible(true);
        comboLegsField.set(contract, comboLegs);
    }

    @Test
    public void testClone() throws Exception {
        // Arrange
        Contract clonedContract = (Contract) contract.clone();
        // Assert
        assertNotSame(contract, clonedContract, "Cloned object should not be the same instance as the original.");
        assertEquals(contract.m_conId, clonedContract.m_conId, "Cloned object should have the same m_conId.");
        assertEquals(contract.m_symbol, clonedContract.m_symbol, "Cloned object should have the same m_symbol.");
        assertEquals(contract.m_secType, clonedContract.m_secType, "Cloned object should have the same m_secType.");
        assertEquals(contract.m_expiry, clonedContract.m_expiry, "Cloned object should have the same m_expiry.");
        assertEquals(contract.m_strike, clonedContract.m_strike, "Cloned object should have the same m_strike.");
        assertEquals(contract.m_right, clonedContract.m_right, "Cloned object should have the same m_right.");
        assertEquals(contract.m_multiplier, clonedContract.m_multiplier, "Cloned object should have the same m_multiplier.");
        assertEquals(contract.m_exchange, clonedContract.m_exchange, "Cloned object should have the same m_exchange.");
        assertEquals(contract.m_currency, clonedContract.m_currency, "Cloned object should have the same m_currency.");
        assertEquals(contract.m_localSymbol, clonedContract.m_localSymbol, "Cloned object should have the same m_localSymbol.");
        assertEquals(contract.m_primaryExch, clonedContract.m_primaryExch, "Cloned object should have the same m_primaryExch.");
        assertEquals(contract.m_includeExpired, clonedContract.m_includeExpired, "Cloned object should have the same m_includeExpired.");
        assertEquals(contract.m_secIdType, clonedContract.m_secIdType, "Cloned object should have the same m_secIdType.");
        assertEquals(contract.m_secId, clonedContract.m_secId, "Cloned object should have the same m_secId.");
        assertEquals(contract.m_comboLegsDescrip, clonedContract.m_comboLegsDescrip, "Cloned object should have the same m_comboLegsDescrip.");
        assertEquals(contract.m_comboLegs, clonedContract.m_comboLegs, "Cloned object should have the same m_comboLegs.");
        assertNotSame(contract.m_comboLegs, clonedContract.m_comboLegs, "Cloned object should have a different m_comboLegs instance.");
        assertEquals(contract.m_underComp, clonedContract.m_underComp, "Cloned object should have the same m_underComp.");
    }
}
