package com.ib.client;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class OrderState_equals_0_1_Test {

    @Test
    public void testEquals() {
        OrderState orderState = new OrderState("open", "0.01", "0.01", "0.01", 0.01, 0.01, 0.01, "USD", "No warning");
        OrderState otherOrderState = new OrderState("open", "0.01", "0.01", "0.01", 0.01, 0.01, 0.01, "USD", "No warning");
        assertEquals(orderState, otherOrderState);
    }
}
