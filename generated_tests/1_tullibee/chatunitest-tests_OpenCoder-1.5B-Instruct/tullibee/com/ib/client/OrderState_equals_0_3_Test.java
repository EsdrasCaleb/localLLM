package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_3_Test {

    private OrderState orderState;

    @Test
    void testEquals() {
        orderState = new OrderState("Active", "0.5", "0.3", "Yes", 100.0, 50.0, 200.0, "USD", "No warnings");
        OrderState otherOrderState = new OrderState("Active", "0.5", "0.3", "Yes", 100.0, 50.0, 200.0, "USD", "No warnings");
        assertTrue(orderState.equals(otherOrderState));
        otherOrderState.m_commission = 150.0;
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_minCommission = 60.0;
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_maxCommission = 220.0;
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_commissionCurrency = "EUR";
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_warningText = "Warning text";
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_status = "Inactive";
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_initMargin = "0.4";
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_maintMargin = "0.2";
        assertFalse(orderState.equals(otherOrderState));
        otherOrderState.m_equityWithLoan = "No";
        assertFalse(orderState.equals(otherOrderState));
    }
}
