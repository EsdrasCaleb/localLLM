package com.ib.client;

import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_3_Test {

    @Test
    public void testEquals() {
        OrderState order1 = new OrderState("Initial State", "Initial Margin", "Maint Margin", "Equity With Loan", 1000.0, 500.0, 2000.0, "USD", "Warning Text");
        OrderState order2 = new OrderState("Initial State", "Initial Margin", "Maint Margin", "Equity With Loan", 1000.0, 500.0, 2000.0, "USD", "Warning Text");
        assertEquals(true, order1.equals(order2));
    }
}
