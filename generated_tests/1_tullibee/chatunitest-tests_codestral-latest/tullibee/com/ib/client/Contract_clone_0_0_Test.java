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
        originalContract = new Contract();
        originalContract.m_comboLegs.add("Leg1");
        originalContract.m_comboLegs.add("Leg2");
    }

    @Test
    void testClone() throws CloneNotSupportedException {
        Contract clonedContract = (Contract) originalContract.clone();
        // Check if the cloned object is not the same instance as the original
        assertNotSame(originalContract, clonedContract);
        // Check if the fields are equal
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
        assertEquals(originalContract.m_comboLegsDescrip, clonedContract.m_comboLegsDescrip);
        assertEquals(originalContract.m_underComp, clonedContract.m_underComp);
        // Check if the m_comboLegs vector is cloned independently
        assertNotSame(originalContract.m_comboLegs, clonedContract.m_comboLegs);
        assertEquals(originalContract.m_comboLegs, clonedContract.m_comboLegs);
    }
}
