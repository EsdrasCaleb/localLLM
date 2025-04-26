package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Order_equals_0_1_Test {

    @Mock
    private Order order1;

    @Mock
    private Order order2;

    @Mock
    private Vector algoParams;

    @InjectMocks
    private Order order3;

    @Test
    public void testEqualsOrderSameInstance() {
        assertTrue(order1.equals(order1));
    }
}
