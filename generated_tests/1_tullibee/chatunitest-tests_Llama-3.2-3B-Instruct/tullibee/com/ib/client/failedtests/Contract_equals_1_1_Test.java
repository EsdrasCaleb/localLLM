package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Contract_equals_1_1_Test {

    @Mock
    private UnderComp underComp;

    @InjectMocks
    private Contract contract;

    @Test
    public void testEquals_SameInstance() {
        assertTrue(contract.equals(contract));
    }

    @Test
    public void testEquals_Null() {
        assertTrue(contract.equals(null));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(contract.equals("not a Contract"));
    }

    @Test
    public void testEquals_DifferentContract() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 100, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        assertFalse(contract.equals(otherContract));
    }

    @Test
    public void testEquals_SameFields() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 100, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        when(contract.m_underComp).thenReturn(underComp);
        when(otherContract.m_underComp).thenReturn(underComp);
        assertTrue(contract.equals(otherContract));
    }

    @Test
    public void testEquals_DifferentComboLegs() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 100, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        Vector comboLegs1 = new Vector();
        Vector comboLegs2 = new Vector();
        comboLegs1.add("leg1");
        comboLegs2.add("leg2");
        when(contract.m_comboLegs).thenReturn(comboLegs1);
        when(otherContract.m_comboLegs).thenReturn(comboLegs2);
        assertFalse(contract.equals(otherContract));
    }

    @Test
    public void testEquals_DifferentStrike() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 101, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        assertFalse(contract.equals(otherContract));
    }

    @Test
    public void testEquals_DifferentRight() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 100, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        when(contract.m_right).thenReturn("CASH");
        when(otherContract.m_right).thenReturn("CASH");
        when(contract.m_exchange).thenReturn("LONDON");
        when(otherContract.m_exchange).thenReturn("LONDON");
        when(contract.m_multiplier).thenReturn("1");
        when(otherContract.m_multiplier).thenReturn("1");
        when(contract.m_localSymbol).thenReturn("ABC");
        when(otherContract.m_localSymbol).thenReturn("ABC");
        when(contract.m_primaryExch).thenReturn("SPX");
        when(otherContract.m_primaryExch).thenReturn("SPX");
        when(contract.m_includeExpired).thenReturn(false);
        when(otherContract.m_includeExpired).thenReturn(false);
        when(contract.m_secIdType).thenReturn("STK");
        when(otherContract.m_secIdType).thenReturn("STK");
        when(contract.m_secId).thenReturn("123456789");
        when(otherContract.m_secId).thenReturn("123456789");
        assertTrue(contract.equals(otherContract));
    }
}
