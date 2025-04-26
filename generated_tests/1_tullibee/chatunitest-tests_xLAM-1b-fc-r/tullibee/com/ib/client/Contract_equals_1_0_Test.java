package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Contract_equals_1_0_Test {

    @Test
    public void testEquals() {
        // Test with same instance
        Contract contract1 = new Contract();
        assertTrue(contract1.equals(contract1));
        // Test with different instance
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        contract2.m_symbol = "AAPL";
        contract2.m_secType = "BOND";
        contract2.m_expiry = "2020-01-01";
        contract2.m_strike = 100.0;
        contract2.m_right = "CALL";
        contract2.m_multiplier = "1.0000";
        contract2.m_exchange = "SMART";
        contract2.m_currency = "USD";
        contract2.m_localSymbol = "AAPL";
        contract2.m_primaryExch = "SMART";
        contract2.m_includeExpired = false;
        contract2.m_secIdType = "Ticker";
        contract2.m_secId = "AAPL";
        assertFalse(contract1.equals(contract2));
        // Test with different values
        contract2.m_conId = 2;
        assertFalse(contract1.equals(contract2));
        contract2.m_secType = "EURIBOR";
        assertFalse(contract1.equals(contract2));
        contract2.m_symbol = "GOOGL";
        assertFalse(contract1.equals(contract2));
        contract2.m_expiry = "2020-02-01";
        assertFalse(contract1.equals(contract2));
        contract2.m_strike = 100.0;
        assertFalse(contract1.equals(contract2));
        contract2.m_right = "PUT";
        assertFalse(contract1.equals(contract2));
        contract2.m_multiplier = "1.0001";
        assertFalse(contract1.equals(contract2));
        contract2.m_exchange = "SMART";
        assertFalse(contract1.equals(contract2));
        contract2.m_currency = "EUR";
        assertFalse(contract1.equals(contract2));
        contract2.m_localSymbol = "GOOGL";
        assertFalse(contract1.equals(contract2));
        contract2.m_primaryExch = "SMART";
        assertFalse(contract1.equals(contract2));
        contract2.m_includeExpired = true;
        assertFalse(contract1.equals(contract2));
        contract2.m_secIdType = "Ticker";
        assertFalse(contract1.equals(contract2));
        contract2.m_secId = "GOOGL";
        assertFalse(contract1.equals(contract2));
        // Test with different combo legs
        contract2.m_comboLegs.add(new ComboLeg());
        assertFalse(contract1.equals(contract2));
        // Test with different underComp
        contract2.m_underComp = new UnderComp();
        assertFalse(contract1.equals(contract2));
    }
}
