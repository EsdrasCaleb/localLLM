package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Contract_clone_0_0_Test {

    private Contract originalContract;

    @BeforeEach
    void setUp() {
        Vector<String> comboLegs = new Vector<>();
        comboLegs.add("Leg1");
        comboLegs.add("Leg2");
        originalContract = new Contract(1, "AAPL", "STK", "2023-12-31", 150.0, "C", "1", "NASDAQ", "USD", "AAPL", comboLegs, "NASDAQ", true, "ISIN", "12345");
    }

    @Test
    void testClone() throws CloneNotSupportedException {
        Contract clonedContract = (Contract) originalContract.clone();
        // Verify that the cloned object is not the same instance as the original
        assertNotSame(originalContract, clonedContract);
        // Verify that the fields are equal
        assertEquals(originalContract.m_conId, clonedContract.m_conId);
        assertEquals(originalContract.m_symbol, clonedContract.m_symbol);
        assertEquals(originalContract.m_secType, clonedContract.m_secType);
        assertEquals(originalContract.m_expiry, clonedContract.m_expiry);
        assertEquals(originalContract.m_strike, clonedContract.m_strike);
        assertEquals(originalContract.m_right, clonedContract.m_right);
        assertEquals(originalContract.m_multiplier, clonedContract.m_multiplier);
        assertEquals(originalContract.m_exchange, clonedContract.m_exchange);
        assertEquals(originalContract.m_currency, clonedContract.m_currency);
        assertEquals(originalContract.m_localSymbol, clonedContract.m_localSymbol);
        assertEquals(originalContract.m_primaryExch, clonedContract.m_primaryExch);
        assertEquals(originalContract.m_includeExpired, clonedContract.m_includeExpired);
        assertEquals(originalContract.m_secIdType, clonedContract.m_secIdType);
        assertEquals(originalContract.m_secId, clonedContract.m_secId);
        // Verify that the comboLegs vector is a different instance
        assertNotSame(originalContract.m_comboLegs, clonedContract.m_comboLegs);
        // Verify that the contents of the comboLegs vector are the same
        assertEquals(originalContract.m_comboLegs, clonedContract.m_comboLegs);
        // Modify the cloned comboLegs to ensure it does not affect the original
        clonedContract.m_comboLegs.add("Leg3");
        assertNotEquals(originalContract.m_comboLegs.size(), clonedContract.m_comboLegs.size());
    }
}
