package com.ib.client;

import java.lang.reflect.Field;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Contract_equals_1_0_Test {

    @Mock
    private UnderComp mockUnderComp1;

    @Mock
    private UnderComp mockUnderComp2;

    private Contract contract1;

    private Contract contract2;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        contract1 = new Contract();
        contract2 = new Contract();
        // Set common fields
        setField(contract1, "m_conId", 1);
        setField(contract2, "m_conId", 1);
        setField(contract1, "m_secType", "STK");
        setField(contract2, "m_secType", "STK");
        setField(contract1, "m_symbol", "AAPL");
        setField(contract2, "m_symbol", "AAPL");
        setField(contract1, "m_exchange", "SMART");
        setField(contract2, "m_exchange", "SMART");
        setField(contract1, "m_primaryExch", "NASDAQ");
        setField(contract2, "m_primaryExch", "NASDAQ");
        setField(contract1, "m_currency", "USD");
        setField(contract2, "m_currency", "USD");
        setField(contract1, "m_strike", 150.0);
        setField(contract2, "m_strike", 150.0);
        setField(contract1, "m_expiry", "20231221");
        setField(contract2, "m_expiry", "20231221");
        setField(contract1, "m_right", "C");
        setField(contract2, "m_right", "C");
        setField(contract1, "m_multiplier", "100");
        setField(contract2, "m_multiplier", "100");
        setField(contract1, "m_localSymbol", "AAPL231221C00150000");
        setField(contract2, "m_localSymbol", "AAPL231221C00150000");
        setField(contract1, "m_secIdType", "ISIN");
        setField(contract2, "m_secIdType", "ISIN");
        setField(contract1, "m_secId", "US0378331005");
        setField(contract2, "m_secId", "US0378331005");
        setField(contract1, "m_comboLegs", new Vector<>());
        setField(contract2, "m_comboLegs", new Vector<>());
        setField(contract1, "m_underComp", mockUnderComp1);
        setField(contract2, "m_underComp", mockUnderComp2);
    }

    private void setField(Object obj, String fieldName, Object value) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }

    @Test
    void testEquals_SameObject() {
        assertTrue(contract1.equals(contract1));
    }

    @Test
    void testEquals_NullObject() {
        assertFalse(contract1.equals(null));
    }

    @Test
    void testEquals_DifferentClass() {
        assertFalse(contract1.equals("Not a Contract"));
    }

    @Test
    void testEquals_DifferentConId() throws Exception {
        setField(contract2, "m_conId", 2);
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentSecType() throws Exception {
        setField(contract2, "m_secType", "OPT");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentSymbol() throws Exception {
        setField(contract2, "m_symbol", "GOOGL");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentExchange() throws Exception {
        setField(contract2, "m_exchange", "CME");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentPrimaryExch() throws Exception {
        setField(contract2, "m_primaryExch", "NYSE");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentCurrency() throws Exception {
        setField(contract2, "m_currency", "EUR");
        assertFalse(contract1.equals(contract2));
    }

    @Test
    void testEquals_DifferentStrike() throws Exception {
        setField(contract2, "m_strike", 160.0);
    }
}
