package com.ib.client;

import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Contract_equals_1_0_Test {

    @Test
    public void testEquals_SameInstance() {
        Contract contract = new Contract();
        assertTrue(contract.equals(contract));
    }

    @Test
    public void testEquals_NullObject() {
        Contract contract = new Contract();
        assertFalse(contract.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        Contract contract = new Contract();
        String differentClassObject = "Not a Contract";
        assertFalse(contract.equals(differentClassObject));
    }

    @Test
    public void testEquals_DifferentConId() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        Contract contract2 = new Contract();
        contract2.m_conId = 2;
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentSecType() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_secType = "STOCK";
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        contract2.m_secType = "BOND";
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_DifferentFields() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_symbol = "AAPL";
        contract1.m_exchange = "NASDAQ";
        contract1.m_primaryExch = "NASDAQ";
        contract1.m_currency = "USD";
        contract1.m_strike = 150.0;
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        contract2.m_symbol = "AAPL";
        // Different exchange
        contract2.m_exchange = "NYSE";
        // Different primary exchange
        contract2.m_primaryExch = "NYSE";
        contract2.m_currency = "USD";
        contract2.m_strike = 150.0;
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_BondContract() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_secType = "BOND";
        contract1.m_expiry = "2025-01-01";
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        contract2.m_secType = "BOND";
        // Different expiry
        contract2.m_expiry = "2026-01-01";
        // Should be true for bond regardless of expiry
        assertTrue(contract1.equals(contract2));
    }

    @Test
    public void testEquals_ComboLegs() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_comboLegs.add("Leg1");
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        // Different combo legs
        contract2.m_comboLegs.add("Leg2");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_UnderComp() {
        UnderComp underComp1 = mock(UnderComp.class);
        UnderComp underComp2 = mock(UnderComp.class);
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_underComp = underComp1;
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        // Different UnderComp
        contract2.m_underComp = underComp2;
        assertFalse(contract1.equals(contract2));
    }

    @Test
    public void testEquals_SameValues() {
        Contract contract1 = new Contract();
        contract1.m_conId = 1;
        contract1.m_symbol = "AAPL";
        contract1.m_secType = "STOCK";
        contract1.m_expiry = "2025-01-01";
        contract1.m_strike = 150.0;
        contract1.m_exchange = "NASDAQ";
        contract1.m_primaryExch = "NASDAQ";
        contract1.m_currency = "USD";
        contract1.m_secIdType = "Type1";
        contract1.m_secId = "ID1";
        contract1.m_comboLegs.add("Leg1");
        contract1.m_underComp = new UnderComp();
        Contract contract2 = new Contract();
        contract2.m_conId = 1;
        contract2.m_symbol = "AAPL";
        contract2.m_secType = "STOCK";
        contract2.m_expiry = "2025-01-01";
        contract2.m_strike = 150.0;
        contract2.m_exchange = "NASDAQ";
    }
}
