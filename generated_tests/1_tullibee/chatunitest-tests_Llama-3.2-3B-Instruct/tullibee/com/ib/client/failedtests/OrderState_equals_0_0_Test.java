package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class OrderState_equals_0_0_Test {

    @Mock
    private OrderState orderStateMock;

    @InjectMocks
    private OrderState orderState;

    @Test
    public void testEquals_SameObject_ReturnsTrue() {
        assertTrue(orderState.equals(orderState));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        assertFalse(orderState.equals(null));
    }

    @Test
    public void testEquals_DifferentObject_ReturnsFalse() {
        when(orderStateMock.m_commission).thenReturn(10.0);
        when(orderStateMock.m_minCommission).thenReturn(10.0);
        when(orderStateMock.m_maxCommission).thenReturn(10.0);
        when(orderStateMock.m_status).thenReturn("status");
        when(orderStateMock.m_initMargin).thenReturn("initMargin");
        when(orderStateMock.m_maintMargin).thenReturn("maintMargin");
        when(orderStateMock.m_equityWithLoan).thenReturn("equityWithLoan");
        when(orderStateMock.m_commissionCurrency).thenReturn("currency");
        OrderState otherOrderState = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 10.0, 10.0, 10.0, "currency", "warningText");
        assertFalse(orderState.equals(otherOrderState));
    }

    @Test
    public void testEquals_SameFields_ReturnsTrue() {
        when(orderStateMock.m_commission).thenReturn(10.0);
        when(orderStateMock.m_minCommission).thenReturn(10.0);
        when(orderStateMock.m_maxCommission).thenReturn(10.0);
        when(orderStateMock.m_status).thenReturn("status");
        when(orderStateMock.m_initMargin).thenReturn("initMargin");
        when(orderStateMock.m_maintMargin).thenReturn("maintMargin");
        when(orderStateMock.m_equityWithLoan).thenReturn("equityWithLoan");
        when(orderStateMock.m_commissionCurrency).thenReturn("currency");
        OrderState otherOrderState = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 10.0, 10.0, 10.0, "currency", "warningText");
        assertTrue(orderState.equals(otherOrderState));
    }
}
