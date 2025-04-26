package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class OrderState_equals_0_0_Test {

    @InjectMocks
    private OrderState orderState;

    @Test
    public void testEquals_SameObject_ReturnsTrue() {
        assertEquals(true, orderState.equals(orderState));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        OrderState other = null;
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_DifferentObjects_ReturnsFalse() {
        OrderState other = new OrderState("differentStatus", "differentInitMargin", "differentMaintMargin", "differentEquityWithLoan", 1.0, 2.0, 3.0, "differentCommissionCurrency", "differentWarningText");
        assertFalse(orderState.equals(other));
    }

    @Test
    public void testEquals_MatchesAllFields_ReturnsTrue() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "commissionCurrency", "warningText");
        assertTrue(orderState.equals(other));
    }

    @Test
    public void testEquals_MismatchedFields_ReturnsFalse() {
        OrderState other = new OrderState("status", "initMargin", "maintMargin", "equityWithLoan", 1.0, 2.0, 3.0, "differentCommissionCurrency", "warningText");
        assertFalse(orderState.equals(other));
    }
}
