package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_0_Test {

    @Test
    public void testEquals_EqualObjects() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        assertEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentObjects() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "EUR", "");
        assertNotEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentStatus() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        assertEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentInitMargin() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "200", 0.0, 0.0, 0.0, "USD", "");
        assertNotEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentMaintMargin() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "200", 0.0, 0.0, 0.0, "USD", "");
        assertNotEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentEquityWithLoan() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "200", 0.0, 0.0, 0.0, "USD", "");
        assertNotEquals(state1, state2);
    }

    @Test
    public void testEquals_DifferentCommissionCurrency() {
        OrderState state1 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "USD", "");
        OrderState state2 = new OrderState("open", "100", "200", "100", 0.0, 0.0, 0.0, "EUR", "");
        assertNotEquals(state1, state2);
    }
}
