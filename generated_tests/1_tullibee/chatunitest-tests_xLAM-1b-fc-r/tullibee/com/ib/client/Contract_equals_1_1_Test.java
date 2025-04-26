package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Contract_equals_1_1_Test {

    @Test
    void testEquals() {
        Contract contract1 = new Contract();
        Contract contract2 = new Contract();
        contract2.m_conId = 123;
        contract2.m_secType = "EQUITY";
        contract2.m_symbol = "AAPL";
        contract2.m_exchange = "NASDAQ";
        contract2.m_primaryExch = "NASDAQ";
        contract2.m_currency = "USD";
        assertTrue(contract1.equals(contract2));
        contract2.m_conId = 456;
        assertFalse(contract1.equals(contract2));
        contract2.m_secType = "FOO";
        assertFalse(contract1.equals(contract2));
        contract2.m_symbol = "BAR";
        assertFalse(contract1.equals(contract2));
        contract2.m_exchange = "FOO";
        assertFalse(contract1.equals(contract2));
        contract2.m_primaryExch = "FOO";
        assertFalse(contract1.equals(contract2));
        contract2.m_currency = "FOO";
        assertFalse(contract1.equals(contract2));
        contract2.m_comboLegs.add(new Object());
        assertFalse(contract1.equals(contract2));
        contract2.m_underComp = new UnderComp();
        assertFalse(contract1.equals(contract2));
    }
}
