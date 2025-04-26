package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    private OrderState orderState1;

    private OrderState orderState2;

    private OrderState orderState3;

    @BeforeEach
    public void setUp() {
        orderState1 = new OrderState("status1", "100", "50", "200", 10.0, 5.0, 15.0, "USD", "No Warning");
        orderState2 = new OrderState("status1", "100", "50", "200", 10.0, 5.0, 15.0, "USD", "No Warning");
        orderState3 = new OrderState("status2", "100", "50", "200", 10.0, 5.0, 15.0, "USD", "No Warning");
    }

    @Test
    public void testEquals_SameObject() {
        assertTrue(orderState1.equals(orderState1));
    }

    @Test
    public void testEquals_NullObject() {
        assertFalse(orderState1.equals(null));
    }

    @Test
    public void testEquals_EqualObjects() {
        assertTrue(orderState1.equals(orderState2));
    }

    @Test
    public void testEquals_DifferentCommission() {
        orderState3.m_commission = 20.0;
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentMinCommission() {
        orderState3.m_minCommission = 10.0;
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentMaxCommission() {
        orderState3.m_maxCommission = 20.0;
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentInitMargin() {
        orderState3.m_initMargin = "200";
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentMaintMargin() {
        orderState3.m_maintMargin = "200";
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentEquityWithLoan() {
        orderState3.m_equityWithLoan = "300";
        assertFalse(orderState1.equals(orderState3));
    }

    @Test
    public void testEquals_DifferentCommissionCurrency() {
        orderState3.m_commissionCurrency = "EUR";
        assertFalse(orderState1.equals(orderState3));
    }
}
