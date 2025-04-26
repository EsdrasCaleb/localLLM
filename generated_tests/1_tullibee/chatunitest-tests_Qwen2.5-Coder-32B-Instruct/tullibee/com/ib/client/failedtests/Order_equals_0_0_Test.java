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
    public void testEquals_DifferentClass_ReturnsFalse() {
        assertFalse(order1.equals(new Object()));
    }

    @Test
    public void testEquals_SamePermId_ReturnsTrue() {
        assertTrue(order1.equals(order2));
    }

    @Test
    public void testEquals_DifferentPermId_ReturnsFalse() {
        order2.m_permId = 2;
        assertFalse(order1.equals(order2));
    }

    @Test
    public void testEquals_DifferentOrderId_ReturnsFalse() {
        order2.m_orderId = 102;
        assertFalse(order1.equals(order2));
    }

    @Test
    public void testEquals_DifferentAction_ReturnsFalse() {
        order2.m_action = "SELL";
        assertFalse(order1.equals(order2));
    }

    @Test
    public void testEquals_DifferentAlgoParams_ReturnsFalse() {
        order1.m_algoParams.add("param1");
        order2.m_algoParams.add("param2");
        assertFalse(order1.equals(order2));
    }

    @Test
    public void testEquals_SameAlgoParams_ReturnsTrue() {
        order1.m_algoParams.add("param1");
        order2.m_algoParams.add("param1");
        assertTrue(order1.equals(order2));
    }

    @Test
    public void testEquals_StringCompareUtil_UsedForStringFields() throws Exception {
        order1.m_orderRef = "ref1";
        order2.m_orderRef = "ref1";
        Field utilField = Util.class.getDeclaredField("instance");
        utilField.setAccessible(true);
        utilField.set(null, mockUtil);
        when(mockUtil.StringCompare(order1.m_orderRef, order2.m_orderRef)).thenReturn(0);
        assertTrue(order1.equals(order2));
        verify(mockUtil).StringCompare(order1.m_orderRef, order2.m_orderRef);
    }

    @Test
    public void testEquals_VectorEqualsUnorderedUtil_UsedForVectorField() throws Exception {
        order1.m_algoParams.add("param1");
        order2.m_algoParams.add("param1");
        Field utilField = Util.class.getDeclaredField("instance");
        utilField.setAccessible(true);
        utilField.set(null, mockUtil);
        when(mockUtil.VectorEqualsUnordered(order1.m_algoParams, order2.m_algoParams)).thenReturn(true);
        assertTrue(order1.equals(order2));
        verify(mockUtil).VectorEqualsUnordered(order1.m_algoParams, order2.m_algoParams);
    }
}
