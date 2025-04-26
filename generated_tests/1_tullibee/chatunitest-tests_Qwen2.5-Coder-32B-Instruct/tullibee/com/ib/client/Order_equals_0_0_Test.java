package com.ib.client;

import java.lang.reflect.Field;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Order_equals_0_0_Test {

    @Mock
    private Util mockUtil;

    private Order order1;

    private Order order2;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        order1 = new Order();
        order2 = new Order();
        // Set some default values to compare
        order1.m_permId = 1;
        order2.m_permId = 1;
        order1.m_orderId = 101;
        order2.m_orderId = 101;
        order1.m_action = "BUY";
        order2.m_action = "BUY";
        order1.m_algoParams = new Vector<>();
        order2.m_algoParams = new Vector<>();
    }

    @Test
    public void testEquals_SameInstance_ReturnsTrue() {
        assertTrue(order1.equals(order1));
    }

    @Test
    public void testEquals_NullObject_ReturnsFalse() {
        assertFalse(order1.equals(null));
    }

    @Test
    public void testEquals_SamePermId_ReturnsTrue() {
        assertTrue(order1.equals(order2));
    }

    @Test
    public void testEquals_SameAlgoParams_ReturnsTrue() {
        order1.m_algoParams.add("param1");
        order2.m_algoParams.add("param1");
        assertTrue(order1.equals(order2));
    }
}
