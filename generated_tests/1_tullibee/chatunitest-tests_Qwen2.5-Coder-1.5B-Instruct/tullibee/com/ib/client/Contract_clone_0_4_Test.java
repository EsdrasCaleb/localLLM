package com.ib.client;

import java.lang.reflect.Constructor;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Contract_clone_0_4_Test {

    @Test
    public void testCloneDeepCopy() throws CloneNotSupportedException {
        // Create a sample contract instance
        Contract original = new Contract(12345, "AAPL", "STK", "20230930", 150.0, "C", "1", "NASDAQ", "USD", "AAPL1", new Vector<>(), "NYSE", true, "CTP", "AAPL");
        // Create a deep copy of the original contract
        Contract cloned = (Contract) original.clone();
        // Check if the cloned contract is not the same as the original
        assertNotSame(original, cloned);
        // Check if the cloned contract has the same properties as the original
        assertEquals(original.m_conId, cloned.m_conId);
        assertEquals(original.m_symbol, cloned.m_symbol);
        assertEquals(original.m_secType, cloned.m_secType);
        assertEquals(original.m_expiry, cloned.m_expiry);
        assertEquals(original.m_strike, cloned.m_strike);
        assertEquals(original.m_right, cloned.m_right);
        assertEquals(original.m_multiplier, cloned.m_multiplier);
        assertEquals(original.m_exchange, cloned.m_exchange);
        assertEquals(original.m_currency, cloned.m_currency);
        assertEquals(original.m_localSymbol, cloned.m_localSymbol);
        assertEquals(original.m_primaryExch, cloned.m_primaryExch);
        assertEquals(original.m_includeExpired, cloned.m_includeExpired);
        assertEquals(original.m_secIdType, cloned.m_secIdType);
        assertEquals(original.m_secId, cloned.m_secId);
        // Check if the cloned contract's combo legs list is not the same as the original
        assertNotSame(original.m_comboLegs, cloned.m_comboLegs);
        // Check if the cloned contract's combo legs list has the same elements as the original
        assertEquals(original.m_comboLegs.size(), cloned.m_comboLegs.size());
        for (int i = 0; i < original.m_comboLegs.size(); i++) {
            assertEquals(original.m_comboLegs.get(i), cloned.m_comboLegs.get(i));
        }
        // Check if the cloned contract's underComp is not the same as the original
        assertNotSame(original.m_underComp, cloned.m_underComp);
        // Check if the cloned contract's underComp has the same properties as the original
        assertEquals(original.m_underComp.getClass(), cloned.m_underComp.getClass());
    }

    @Test
    public void testCloneThrowsCloneNotSupportedException() {
        // Attempting to clone a null object should throw CloneNotSupportedException
        assertThrows(CloneNotSupportedException.class, () -> ((Contract) null).clone());
    }
}
